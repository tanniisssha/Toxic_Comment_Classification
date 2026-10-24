# FairTox AI Streamlit App

1. Place `app.py` and `requirements.txt` in your project folder.
2. Add `final_model/`, `tokenizer/`, and `toxic_preprocessing.py` beside `app.py`.
3. Copy Member 3 and Member 4 output folders into `member3_outputs/` and `member4_outputs/`.
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `streamlit run app.py`

The app reads actual saved model and report files when present. Some dashboard sections remain empty until corresponding member outputs are added. Confirm the CSV filenames/column names in your team's exports; update the candidate paths in `app.py` if they differ.
