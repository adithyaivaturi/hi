import face_recognition
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine
import pyodbc
from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import List
import pandas as pd
from tempfile import NamedTemporaryFile
import shutil
import os

# Define the connection string for SQL Server using pyodbc
conn_str = (
    r'DRIVER={SQL Server};'
    r'SERVER=DESKTOP-QCOS2KB\SQLEXPRESS;'
    r'DATABASE=python project;'
    r'Trusted_Connection=yes;'
)

# Create SQLAlchemy engine
engine = create_engine('mssql+pyodbc://', creator=lambda: pyodbc.connect(conn_str))

app = FastAPI()

@app.get("/compare_faces")
async def compare_faces(file: UploadFile = File(...)):
    try:
        with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        unknown_image = face_recognition.load_image_file(tmp_path)
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        data = get_face_recognition_data()

        match_found = False
        match_details = {"match": False, "message": "These are different people.", "MCode": "", "MName": ""}

        for i in data:
            known_image = face_recognition.load_image_file(i['Photo'])
            known_encoding = face_recognition.face_encodings(known_image)[0]
            results = face_recognition.compare_faces([known_encoding], unknown_encoding)

            if results[0]:
                match_details = {
                    "match": True,
                    "message": "These are the same person.",
                    "MCode": i["MCode"],
                    "MName": i["MName"]
                }
                match_found = True
                break  # Exit the loop once a match is found

        return JSONResponse(content=match_details)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'tmp_path' in locals():
            try:
                os.remove(tmp_path)
            except Exception as e:
                pass

def get_face_recognition_data():
    try:
        query = "SELECT * FROM Face_Recognition"
        df = pd.read_sql(query, con=engine)
        return df.to_dict(orient='records')  # Convert DataFrame to list of dictionaries (JSON serializable)
    except Exception as e:
        return {"error": str(e)}
    finally:
        engine.dispose()  # Dispose the engine
