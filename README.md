Terminal 

python -m venv dashboard_env

dashboard_env\Scripts\activate


Struktur File di VS Code Explorer
dashboard_depresi/
dashboard_env/          # Virtual environment
app.py                  # Dashboard code
requirements.txt        # Dependencies
student_depression_dataset.csv  # Dataset

# Install semua requirements sekaligus
pip install -r requirements.txt

di terminal promt masuk env sampe (dashboard_env) C:\path\to\dashboard_depresi>
baru run bawahe
streamlit run app.py

