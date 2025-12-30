# 🚀 Streamlit Community Cloud Dağıtım Rehberi

Bu proje "Local-First" tasarlandığı için (Ollama, Local Neo4j), Streamlit Cloud'a yüklerken **Cloud Moduna** (OpenAI + Neo4j Aura) geçiş yapmanız gerekir.

### 1. Hazırlık
Projenin GitHub reposunda şu iki dosyanın olduğundan emin olun (Ben ekledim):
- `packages.txt` (İçinde `ffmpeg` yazmalı)
- `requirements.txt` (Tüm kütüphaneler)

### 2. Deployment Adımları
1.  [share.streamlit.io](https://share.streamlit.io/) adresine gidin.
2.  GitHub hesabınızı bağlayın ve repository'nizi seçin.
3.  **Deploy** butonuna basmadan önce **"Advanced Settings"** -> **"Secrets"** bölümünü açın.

### 3. Secrets (Gizli Anahtarlar) Ayarı
Aşağıdaki konfigürasyonu Secrets alanına yapıştırın ve kendi değerlerinizle doldurun.
**Önemli:** Cloud ortamında `USE_LOCAL_LLM = False` olmalıdır!

```toml
# --- APP CONFIG ---
USE_LOCAL_LLM = false
LOCAL_LLM_MODEL = "mistral"

# --- OPENAI API (Cloud Modu İçin Şart) ---
OPENAI_API_KEY = "sk-proj-..."

# --- NEO4J AURA (Cloud GraphDB) ---
# Burası localhost olamaz! Ücretsiz Neo4j Aura hesabı açın: neo4j.com/cloud/aura
NEO4J_URI = "neo4j+s://xxxxxxxx.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "sizin-aura-sifreniz"
```

### 4. Neden Local Mod Çalışmaz?
Streamlit Cloud sunucularında `Ollama` çalıştıramazsınız ve sunucu sizin bilgisayarınızdaki (localhost) Neo4j'e erişemez. Bu yüzden veritabanını internete (Neo4j Aura), LLM'i de API'a (OpenAI) taşımanız gerekir.
