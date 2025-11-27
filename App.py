import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path
import requests
import pyodbc
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import torch.nn.functional as F
import pandas as pd

class DatabaseManager:
    def __init__(self, server, database, username, password, driver):
        self.connection_string = (
            f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'
        )

    def get_connection(self):
        return pyodbc.connect(self.connection_string)

    def save_result(self, predicted_class: str, probability: float, location_: str):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO AirQualityResults (predicted_class, probability, location_)
                VALUES (?, ?, ?)
            """, (predicted_class, float(probability), location_))
            conn.commit()
            conn.close()
        except Exception as e:
            print("помилка")
            
    def get_results(self):
        try:
            conn = self.get_connection()
            
            query = """
                SELECT predicted_class, probability, location_, date_time
                FROM AirQualityResults
                ORDER BY date_time DESC
            """
            

            df = pd.read_sql(query, conn)
            conn.close()
            return df
        
        except Exception as e:
            print(f"Помилка отримання випадкових результатів: {e}")
            return pd.DataFrame(columns=[
                'predicted_class', 'probability', 'location_', 'date_time'
            ])
            
    def get_count_types(self):

        class_order = [
            'Добре',
            'Помірне',
            'Шкідливе для чутливих груп',
            'Шкідливе',
            'Дуже шкідливе'
        ]
        
        try:
            conn = self.get_connection()
            
            query = """
                SELECT
                    predicted_class,
                    COUNT(*) AS count_of_class
                FROM
                    AirQualityResults
                GROUP BY
                    predicted_class;
            """
            
            df = pd.read_sql(query, conn, index_col='predicted_class')
            conn.close()

            final_counts = df.reindex(class_order, fill_value=0)['count_of_class']

            return final_counts.to_list()

        except Exception as e:
            print(f"Помилка отримання підрахунків за типами: {e}")
            return [0, 0, 0, 0, 0]
        


MODEL_NAME = 'google/vit-base-patch16-224'
NUM_LABELS = 5
SAVE_PATH = 'models/vit_air_quality_model.pth'


device = torch.device("cpu") 

class ModelManager:
    
    @staticmethod
    @st.cache_resource
    def load_model_cached(_self=None):
        try:
            processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
            

            model = ViTForImageClassification.from_pretrained(
                MODEL_NAME,
                num_labels=NUM_LABELS,
                ignore_mismatched_sizes=True 
            )
            
            model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
            
            model.eval()
            
            model.to(device)
            return model, processor 

        except Exception as e:
            print(f"Помилка завантаження моделі ViT: {e}")
            return None, None

    def predict(self, model, processor, image: Image.Image):
        if model is None or processor is None:
            st.error("Модель або процесор не завантажено.")
            return None
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        return probabilities.cpu().numpy()




class AirQualityInfo:
    def __init__(self):
        self.DIR_ = Path('details_air_quality')
        self.mapping = {
            0: 'good.txt',
            1: 'moderate.txt',
            2: 'unh_for_sens.txt',
            3: 'unhealthy.txt',
            4: 'very_unh.txt'
        }

    def get_info(self, number: int):
        file_path = self.DIR_ / self.mapping[number]
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()



class AirQualityApp:
    def __init__(self):
        self.class_names = [
            'Добре',
            'Помірне',
            'Шкідливе для чутливих груп',
            'Шкідливе',
            'Дуже шкідливе'
        ]

        self.db = DatabaseManager(
            server='VIVIBOOK\\SQLEXPRESS',
            database='Air_quality',
            username='sa',
            password='123456',
            driver='{ODBC Driver 17 for SQL Server}'
        )
        self.model_manager = ModelManager()
        
        self.info_manager = AirQualityInfo()
        self.model, self.processor = self.model_manager.load_model_cached()
        

    def get_user_location(self):
        try:
            ip_data = requests.get("https://api64.ipify.org?format=json").json()
            user_ip = ip_data["ip"]
            geo = requests.get(f"https://ipinfo.io/{user_ip}/json").json()
            return f"{geo.get('city', '')}, {geo.get('region', '')}, {geo.get('country', '')}"
        except Exception:
            return "Невідоме місце розташування"

    def run(self):
        tab1, tab2, tab3= st.tabs(["Головна","Аналітика","Карта якості повітря"])
        
        with tab2:
            st.title("Ви можете ознайомитись з історичними даними")
            random_df = self.db.get_results()
            
            if random_df.empty:
                st.warning("Не вдалося отримати випадкові записи або база порожня.")
            else:
                random_df_display = random_df.copy()
                random_df_display['probability'] = random_df_display['probability'].map('{:,.2f}%'.format)
                random_df_display['date_time'] = random_df_display['date_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                random_df_display.columns = ['Клас', 'Впевненість', 'Локація', 'Час аналізу']
                
                st.dataframe(random_df_display, use_container_width=True, hide_index=True)
            st.title("Візуалізація забруднень в діаграмі")
            counts = self.db.get_count_types()
            names = ["Добре", "Помірне", "Шкідливе для чутливих груп", "Шкідливе","Дуже шкідливе"]
            df = pd.DataFrame(data = {"Значення" : counts}, index = names)
            st.bar_chart(df)
            
            
        with tab3:
            st.title("Карта якості повітря (IQAir)")

            embed_url = "https://www.iqair.com/air-quality-map?lat=49.4216&lng=26.9965&zoomLevel=10" 
            import streamlit.components.v1 as components

            components.html(
            f"""
            <iframe
            src="{embed_url}"
            width="100%"
            height="700"
            style="border: none;"
            allowfullscreen
            loading="lazy"
            ></iframe>
            """,
            height=720,
            )
        
        with tab1:
            st.title("Визначення якості повітря за фото")
            uploaded_file = st.file_uploader("Оберіть фото для аналізу:", type=['jpg', 'jpeg'])

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Завантажене фото", use_container_width=True)

                if st.button("Дізнатись рівень забруднення"):

                    predictions = self.model_manager.predict(self.model, self.processor, image)
                    number = np.argmax(predictions)
                    predicted_class = self.class_names[number]
                    confidence = np.max(predictions) * 100

                    location = self.get_user_location()
                    if number > 2:
                        st.error(f"Рівень якості повітря: {predicted_class} ({confidence:.2f}%)")
                    elif number == 2:
                        st.warning(f"Рівень якості повітря: {predicted_class} ({confidence:.2f}%)")
                    else:
                        st.success(f"Рівень якості повітря: {predicted_class} ({confidence:.2f}%)")

                    self.db.save_result(predicted_class, confidence, location)

                    st.info(self.info_manager.get_info(number))



if __name__ == "__main__":
    app = AirQualityApp()
    app.run()

#python -m streamlit run App.py