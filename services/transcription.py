"""
Transcription Service for the Podcast Knowledge Graph System.
Handles YouTube audio download and AssemblyAI transcription with speaker diarization.
"""

import os
import asyncio
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

import assemblyai as aai
import yt_dlp

from models.entities import TranscriptSegment, Episode
from config import get_settings, get_logger

logger = get_logger(__name__)


class TranscriptionService:
    """
    Service for downloading YouTube audio and transcribing with speaker diarization.
    
    Features:
    - YouTube audio download via yt-dlp
    - AssemblyAI transcription with speaker diarization
    - Transcript caching to avoid re-processing
    - Speaker identification and labeling
    - Progress tracking
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache_dir = Path(self.settings.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure AssemblyAI
        aai.settings.api_key = self.settings.assemblyai_api_key
        self.transcriber = aai.Transcriber()
    
    async def download_youtube_audio(self, video_id: str) -> str:
        """
        Download audio from a YouTube video.
        
        Args:
            video_id: YouTube video ID (e.g., 'd6EMk6dyrOU')
            
        Returns:
            Path to the downloaded audio file
        """
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        output_path = self.cache_dir / f"{video_id}.mp3"
        
        # Check if already downloaded
        if output_path.exists():
            logger.info(f"Audio already cached: {output_path}")
            return str(output_path)
        
        logger.info(f"Downloading audio from: {video_url}")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(self.cache_dir / f"{video_id}.%(ext)s"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._download_with_ytdlp(video_url, ydl_opts)
            )
            
            logger.info(f"Audio downloaded: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to download audio: {str(e)}")
            raise
    
    def _download_with_ytdlp(self, url: str, opts: dict) -> None:
        """Synchronous yt-dlp download."""
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    
    async def get_video_info(self, video_id: str) -> Dict[str, Any]:
        """
        Get video metadata from YouTube.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary with video metadata
        """
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        try:
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(
                None,
                lambda: self._extract_info(video_url, ydl_opts)
            )
            
            return {
                'title': info.get('title', ''),
                'duration': info.get('duration', 0),
                'upload_date': self._format_date(info.get('upload_date', '')),
                'channel': info.get('channel', ''),
                'description': info.get('description', ''),
                'view_count': info.get('view_count', 0),
            }
            
        except Exception as e:
            logger.error(f"Failed to get video info: {str(e)}")
            return {}
    
    def _extract_info(self, url: str, opts: dict) -> dict:
        """Synchronous info extraction."""
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)
    
    def _format_date(self, date_str: str) -> str:
        """Format date from YYYYMMDD to YYYY-MM-DD."""
        if len(date_str) == 8:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        return date_str
    
    async def transcribe_with_diarization(
        self,
        audio_path: str,
        speakers_expected: Optional[int] = None
    ) -> List[TranscriptSegment]:
        """
        Transcribe audio using either AssemblyAI (Cloud) or Whisper (Local).
        """
        # Check for cached transcript
        cache_key = self._get_cache_key(audio_path)
        cached = self._load_cached_transcript(cache_key)
        if cached:
            logger.info(f"Using cached transcript: {cache_key}")
            return cached
        
        if self.settings.use_local_llm:
            return await self._transcribe_local_whisper(audio_path, cache_key)
        else:
            return await self._transcribe_assemblyai(audio_path, speakers_expected, cache_key)

    async def _transcribe_local_whisper(self, audio_path: str, cache_key: str) -> List[TranscriptSegment]:
        """Transcribe using local Whisper model."""
        import whisper
        import torch
        
        logger.info(f"Transcribing locally with Whisper ({self.settings.local_whisper_model})...")
        
        try:
            # Load model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # For Mac M1/M2, use 'cpu' or 'mps' (Whisper supports cpu well)
            if torch.backends.mps.is_available():
                device = "mps"
                
            model = whisper.load_model(self.settings.local_whisper_model, device=device)
            print(f"Whisper model loaded on {device}")
            
            # Transcribe
            # Run in executor to avoid blocking async loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: model.transcribe(audio_path, fp16=False) # fp16=False for CPU compatibility
            )
            
            segments = []
            for seg in result["segments"]:
                segments.append(TranscriptSegment(
                    text=seg["text"].strip(),
                    start=seg["start"],
                    end=seg["end"],
                    speaker="Speaker",  # Whisper base doesn't distinguish speakers
                    confidence=1.0  # Whisper doesn't give segment confidnece easily
                ))
            
            self._cache_transcript(cache_key, segments)
            logger.info(f"Local transcription complete: {len(segments)} segments")
            return segments
            
        except Exception as e:
            logger.error(f"Local Whisper transcription failed: {e}")
            raise

    async def _transcribe_assemblyai(self, audio_path: str, speakers_expected: int, cache_key: str) -> List[TranscriptSegment]:
        """Transcribe using AssemblyAI API."""
        logger.info(f"Transcribing audio with diarization (AssemblyAI): {audio_path}")
        
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            speakers_expected=speakers_expected,
            language_code="en",
            punctuate=True,
            format_text=True,
        )
        
        try:
            loop = asyncio.get_event_loop()
            transcript = await loop.run_in_executor(
                None,
                lambda: self.transcriber.transcribe(audio_path, config=config)
            )
            
            if transcript.status == aai.TranscriptStatus.error:
                raise Exception(f"Transcription failed: {transcript.error}")
            
            segments = self._parse_transcript(transcript)
            self._cache_transcript(cache_key, segments)
            return segments
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise
    
    def _parse_transcript(self, transcript: aai.Transcript) -> List[TranscriptSegment]:
        """Parse AssemblyAI transcript into TranscriptSegment objects."""
        segments = []
        
        if transcript.utterances:
            # Use utterances if speaker diarization is available
            for utterance in transcript.utterances:
                segment = TranscriptSegment(
                    text=utterance.text,
                    start=utterance.start / 1000.0,  # Convert ms to seconds
                    end=utterance.end / 1000.0,
                    speaker=f"Speaker {utterance.speaker}",
                    confidence=utterance.confidence or 0.9
                )
                segments.append(segment)
        elif transcript.words:
            # Fallback to word-level if no utterances
            current_text = []
            current_start = 0
            
            for i, word in enumerate(transcript.words):
                if i == 0:
                    current_start = word.start / 1000.0
                
                current_text.append(word.text)
                
                # Create segment every ~30 words or at sentence end
                if len(current_text) >= 30 or (word.text and word.text[-1] in '.!?'):
                    segment = TranscriptSegment(
                        text=" ".join(current_text),
                        start=current_start,
                        end=word.end / 1000.0,
                        speaker=None,
                        confidence=word.confidence or 0.9
                    )
                    segments.append(segment)
                    current_text = []
                    current_start = word.end / 1000.0
            
            # Add remaining text
            if current_text:
                segment = TranscriptSegment(
                    text=" ".join(current_text),
                    start=current_start,
                    end=transcript.words[-1].end / 1000.0,
                    speaker=None,
                    confidence=0.9
                )
                segments.append(segment)
        
        return segments
    
    def identify_speakers(
        self,
        segments: List[TranscriptSegment],
        hosts: List[str],
        guests: List[str] = None
    ) -> List[TranscriptSegment]:
        """
        Attempt to identify speakers based on context.
        
        This is a heuristic approach - ideally you'd use voice recognition
        or manual labeling for accurate results.
        
        Args:
            segments: List of transcript segments
            hosts: Known host names
            guests: Known guest names
            
        Returns:
            Segments with updated speaker labels
        """
        if not hosts:
            return segments
        
        all_speakers = hosts + (guests or [])
        speaker_mapping = {}
        
        # Analyze speaking patterns to map Speaker A, B, etc. to names
        # This is a simplified heuristic
        for segment in segments[:20]:  # Look at first 20 segments
            text_lower = segment.text.lower()
            for person in all_speakers:
                # Check if the person introduces themselves
                name_lower = person.split()[0].lower() if person else ""
                if f"i'm {name_lower}" in text_lower or f"my name is {name_lower}" in text_lower:
                    if segment.speaker and segment.speaker not in speaker_mapping:
                        speaker_mapping[segment.speaker] = person
        
        # If we found mappings, apply them
        if speaker_mapping:
            updated_segments = []
            for segment in segments:
                if segment.speaker in speaker_mapping:
                    updated_segment = TranscriptSegment(
                        text=segment.text,
                        start=segment.start,
                        end=segment.end,
                        speaker=speaker_mapping[segment.speaker],
                        confidence=segment.confidence
                    )
                    updated_segments.append(updated_segment)
                else:
                    updated_segments.append(segment)
            return updated_segments
        
        # Default: assign hosts to first speaker(s)
        if len(hosts) == 1:
            speaker_mapping["Speaker A"] = hosts[0]
        
        return segments
    
    def _get_cache_key(self, audio_path: str) -> str:
        """Generate cache key from audio file."""
        file_hash = hashlib.md5(audio_path.encode()).hexdigest()[:12]
        return f"transcript_{file_hash}"
    
    def _load_cached_transcript(self, cache_key: str) -> Optional[List[TranscriptSegment]]:
        """Load cached transcript if available."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                return [TranscriptSegment(**item) for item in data]
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None
    
    def _cache_transcript(self, cache_key: str, segments: List[TranscriptSegment]) -> None:
        """Cache transcript to disk."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump([s.to_dict() for s in segments], f)
            logger.debug(f"Cached transcript: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache transcript: {e}")
    
    def cleanup_audio(self, video_id: str) -> None:
        """Remove downloaded audio file."""
        audio_path = self.cache_dir / f"{video_id}.mp3"
        if audio_path.exists():
            audio_path.unlink()
            logger.info(f"Cleaned up audio: {audio_path}")


# Convenience functions for simpler usage
async def download_youtube_audio(video_id: str) -> str:
    """Download YouTube audio (convenience function)."""
    service = TranscriptionService()
    return await service.download_youtube_audio(video_id)


async def transcribe_with_diarization(audio_path: str) -> List[TranscriptSegment]:
    """Transcribe audio with diarization (convenience function)."""
    service = TranscriptionService()
    return await service.transcribe_with_diarization(audio_path)


def identify_speakers(
    segments: List[TranscriptSegment],
    hosts: List[str]
) -> List[TranscriptSegment]:
    """Identify speakers (convenience function)."""
    service = TranscriptionService()
    return service.identify_speakers(segments, hosts)
