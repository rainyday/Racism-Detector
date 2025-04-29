# Racism-Detector
Racism Detector adalah sebuah proyek machine learning yang dapat menganalisis teks dan mendeteksi apakah mengandung konten rasis atau tidak. Alat ini menggunakan model Logistic Regression dengan ekstraksi fitur TF-IDF untuk mengklasifikasikan teks.

Proyek ini dapat digunakan untuk:
    1. Memoderasi konten di platform media sosial
    2. Menganalisis komentar atau pesan untuk mendeteksi ujaran kebencian
    3. Sebagai alat bantu dalam penelitian linguistik atau sosial

Langkah - Langkah menjalankan projek:

Langkah 1: Clone Repository GitHub
    Buka terminal/CMD (Windows) atau terminal (Linux/Mac).
    Jalankan perintah berikut untuk mengunduh proyek:
    
    git clone https://github.com/rainyday/Racism-Detector.git
    cd Racism-Detector
Langkah 2: Persiapan Environment
A. Buat Virtual Environment (Opsional)

    python -m venv venv

B. Aktifkan Virtual Environment:
Windows:
    
    venv\Scripts\activate
Linux/Mac:

    source venv/bin/activate

C.Install Dependencies

    pip install pandas scikit-learn nltk

Langkah 3: Download Data NLTK

    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

Langkah 4: Jalankan Program
Vs Code : 
    
    code .python SC.py
    
