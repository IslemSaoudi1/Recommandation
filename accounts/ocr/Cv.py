import os
import cv2
import pytesseract
import pandas as pd
import numpy as np
import re
import pyodbc
import configparser
import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
from ultralytics import YOLO
nlp = spacy.load('fr_core_news_sm')
import nltk
from .Skills import technical_skills_list
from .Profile import profil_list


def replace_empty_with_none(df):
    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Check if the column has empty values
        if df[column].empty or df[column].dtype == None:
            # Replace empty values with None
            df[column] = df[column].replace(np.nan, None)
    return df


class CVAnalyzer:
    def __init__(self):
        self.model = None
        self.nlp = spacy.load("fr_core_news_sm")

    def load_yolo_model(self, model_path):
        self.model = YOLO(model_path)

    def get_predictions(self, image_path, confidence):
        img = cv2.imread(image_path)
        results = self.model.predict(source=img, conf=confidence)
        return results

    def extract_values_from_image(self, img_path, save_path, name, bboxes, probs, names):
        img = cv2.imread(img_path)
        img2 = img.copy()
        class_names = {
            0: 'profile',
            1: 'competance',
            2: 'experience_professionnelle',
            3: 'formation',
            4: 'langues',
            5: 'centre',
            6: 'Contact'
        }
        dicts = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}}
        df = pd.DataFrame(
            columns=['profile', 'competance', 'experience_professionnelle', 'formation', 'langues', 'centre',
                     'Contact'])

        for box, prob, index in zip(bboxes, probs.tolist(), names):
            class_dict = dicts[int(index)]
            class_dict[prob] = box

        for index, class_dict in dicts.items():
            if len(class_dict) != 0:
                max_prob = max(class_dict.keys())
                box = class_dict[max_prob]
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cropped_image = img2[y1:y2, x1:x2]
                folder_path = class_names[index]
                if not os.path.exists(os.path.join(save_path, folder_path)):
                    os.makedirs(os.path.join(save_path, folder_path))
                file_path = os.path.join(save_path, folder_path, f"{name.split('.')[0]}-{index}.jpg")
                cv2.imwrite(file_path, cropped_image)
                print(file_path)
                text = pytesseract.image_to_string(cropped_image)
                df.at[0, folder_path] = text if text else None

        df.to_csv('accounts/ocr/my_dataframe.csv', index=False)
        return df

    def clean_and_sort_dataframe(self,csv_file):
        df = pd.read_csv(csv_file)
        df = df.apply(lambda x: x.astype(str).str.lower().str.strip() if x.dtype == "object" else x)
        df = df.replace({'\n': ' ', '\r': ' '}, regex=True)
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].sort_values()
        print(df)
        return df

    def extract_technical_skills(self, text):
        doc = nlp(text)
        matcher = PhraseMatcher(nlp.vocab)
        skills = []

        # Create patterns for each technical skill
        patterns = [nlp(skill.lower()) for skill in technical_skills_list]

        # Add patterns to the matcher
        matcher.add("TechnicalSkills", None, *patterns)

        # Find matches in the text
        matches = matcher(doc)

        # Extract technical skills
        for match_id, start, end in matches:
            skills.append(doc[start:end].text)

        if skills:
            return list(set(skills))
        else:
            return None

    def add_technical_skills(self, df):

        df['competance'] = df['competance'].astype(str).str.lower()
        # Convert 'competance' column to string
        df['competance'] = df['competance'].apply(self.extract_technical_skills)
        return df

    def extract_languages_from_dataframe(self, df, language_keywords):
        for index, row in df.iterrows():
            text = row['langues']
            languages = re.findall(r"\b([A-Za-zéèëêàâîïôùûü]+)\b", text)
            normalized_languages = []
            for language in languages:
                for keyword in language_keywords:
                    if keyword.lower() in language.lower():
                        normalized_languages.append(keyword)
            df.at[index, 'langues'] = ', '.join(normalized_languages) if normalized_languages else None
        print(df)

    def extract_contact_info(self, df):
        # Définition des expressions régulières
        phone_regex = r"\b\d{2}\s?\d{3}\s?\d{3}\b|\b\d{8}\b"
        email_regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+"

        for index, row in df.iterrows():
            # Texte d'origine dans df['Contact']
            text = row['Contact']
            # Extraction du numéro de téléphone
            phone_number = re.search(phone_regex, text)
            phone_number = phone_number.group() if phone_number else ""
            df.at[index, 'phone number'] = phone_number

            # Extraction de l'email
            email = re.search(email_regex, text)
            email = email.group() if email else ""
            df.at[index, 'email'] = email

            # Extraction du profil LinkedIn
            linkedin_profile = ""
            start_index = text.find("linkedin.com/in/")
            if start_index != -1:
                end_index = text.find(" ", start_index)
                if end_index == -1:
                    end_index = len(text)
                linkedin_profile = text[start_index:end_index]
            df.at[index, 'linkedin'] = linkedin_profile
        return df

    def extract_profile(self, text):
        doc = self.nlp(text)
        matcher = PhraseMatcher(self.nlp.vocab)
        profiles = []

        # Create patterns for each profile
        patterns = [self.nlp(profile.lower()) for profile in profil_list]

        # Add patterns to the matcher
        matcher.add("Profiles", None, *patterns)

        # Find matches in the text
        matches = matcher(doc)

        # Extract profiles
        for match_id, start, end in matches:
            profiles.append(doc[start:end].text)

        if profiles:
            return profiles[0]  # Return the first profile as a string
        else:
            return None

    def apply_extraction(self, df):
        df["extracted_profile"] = df["profile"].apply(self.extract_profile)
        return df

    def replace_empty_with_none(self, df):
        # Iterate over each column in the DataFrame
        for column in df.columns:
            # Check if the column has empty values
            if df[column].empty or df[column].dtype == None:
                # Replace empty values with None
                df[column] = df[column].replace(np.nan, None)
        return df

    def replace(self, df, output_file):
        new_df = self.replace_empty_with_none(df)
        new_df.to_csv(output_file, index=False)
        return new_df

    def save_in_database(self, df):
        # Load the config file
        config = configparser.ConfigParser()
        config.read('config.ini')
        cnxn_table = (
            "Driver=ODBC Driver 17 for SQL Server;"
            "Server=ISLEM;"
            "Database=job_finder;"
            "Trusted_Connection=yes;")
        print(cnxn_table)
        # Establish a database connection
        connection_table = pyodbc.connect(cnxn_table)
        print("successfully")
        # Connexion à la base de données
        connection_table = pyodbc.connect(cnxn_table)

        # Activation de l'autocommit pour valider automatiquement les transactions
        connection_table.autocommit = True
        df.columns
        # Création d'un curseur pour exécuter les commandes SQL
        cursor = connection_table.cursor()
        # Itération sur les lignes de la DataFrame et insertion de nouvelles lignes dans la table
        for index, row in df.iterrows():
            extracted_profile = str(row['extracted_profile'])
            profile = str(row['profile'])

            if extracted_profile and extracted_profile != "None":
                Profil = extracted_profile
            else:
                Profil = profile

            Competences = str(row['competance'])
            Experiences_Professionnelles = str(row['experience_professionnelle'])
            Formation = str(row['formation'])
            Langues = str(row['langues'])
            Centre = str(row['centre'])
            Contact = str(row['Contact'])
            Phone_Number = str(row['phone number'])  # Corrected column name with space
            Email = str(row['email'])
            LinkedIn = str(row['linkedin'])

            # Insertion d'une nouvelle ligne dans la table CV avec toutes les colonnes
            cursor.execute(
                'INSERT INTO CV (Profil, Competences, Experiences_Professionnelles, Formation, Langues,Centre, Contact, Phone_Number, Email, LinkedIn) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (Profil, Competences, Experiences_Professionnelles, Formation, Langues, Centre, Contact,
                 Phone_Number, Email, LinkedIn))

        # Validation de la transaction et fermeture du curseur et de la connexion
        connection_table.commit()
        cursor.close()
        connection_table.close()


# Configuration
