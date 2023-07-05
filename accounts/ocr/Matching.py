import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pyodbc
import configparser


class JobMatching:
    def __init__(self, job_offers_file, cv_file, config_file):
        self.job_offers_file = job_offers_file
        self.cv_file = cv_file
        self.config_file = config_file
        self.job_offers_df = None
        self.cv_data_df = None
        self.tokenizer = None
        self.model = None
        self.connection_table = None
        self.cursor_table = None

    def load_data(self):
        # Load the job offers data from the CSV file
        self.job_offers_df = pd.read_csv(self.job_offers_file)
        # Load the CV data from the CSV file
        self.cv_data_df = pd.read_csv(self.cv_file)

    def initialize_model(self):
        # Initialize the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained('camembert-base')
        self.model = AutoModel.from_pretrained('camembert-base')

    def calculate_similarity(self, job_offer_sentence, cv_sentence):
        encoded_job_offer = self.tokenizer(str(job_offer_sentence), truncation=True, padding=True, return_tensors='pt')
        encoded_cv_sentence = self.tokenizer(str(cv_sentence), truncation=True, padding=True, return_tensors='pt')

        job_offer_embedding = self.model(**encoded_job_offer).last_hidden_state.mean(dim=1)
        cv_embedding = self.model(**encoded_cv_sentence).last_hidden_state.mean(dim=1)

        similarity_score = cosine_similarity(job_offer_embedding.detach().numpy(), cv_embedding.detach().numpy())[0][0]

        return similarity_score

    def process_matching(self):
        # Create a DataFrame to store the results

        top_jobs = []
        # Iterate over each CV and calculate similarity scores for each job offer
        for cv_index in range(len(self.cv_data_df)):
            cv_data = self.cv_data_df.loc[cv_index, 'competance']
            similarity_scores = []


            if not pd.isnull(cv_data):  # If competence column is not empty
                for index, job_offer_row in self.job_offers_df.iterrows():
                    job_offer_description = job_offer_row['technical_skills']
                    similarity_score = self.calculate_similarity(job_offer_description, cv_data)
                    similarity_scores.append((index, similarity_score))
            else:
                for index, job_offer_row in self.job_offers_df.iterrows():
                    job_offer_title = job_offer_row['clean_job_title']
                    job_offer_description = self.cv_data_df.loc[cv_index, 'profile']
                    similarity_score = self.calculate_similarity(job_offer_description, job_offer_title)
                    similarity_scores.append((index, similarity_score))

            similarity_scores.sort(key=lambda x: x[1], reverse=True)

            # Store the top 3 job offers in the results DataFrame
            for i in range(min(3, len(similarity_scores))):
                job_offer_index = similarity_scores[i][0]
                similarity_score = similarity_scores[i][1]
                job_offer_title = self.job_offers_df.loc[job_offer_index, 'clean_job_title']
                top_jobs.append({
                    'CV': cv_index + 1,
                    'Job_Offer': job_offer_index + 1,
                    'Job_Title': job_offer_title,
                    'Similarity_Score': similarity_score
                })

        results_df = pd.DataFrame(top_jobs, columns=['CV', 'Job_Offer', 'Job_Title', 'Similarity_Score'])

        # Save the results to a CSV file
        results_df.to_csv('accounts/ocr/results.csv', index=False)

        return top_jobs
    def Save_data(self):
        # Establish a database connection
        config = configparser.ConfigParser()
        config.read(self.config_file)
        cnxn_table = (
            "Driver=ODBC Driver 17 for SQL Server;"
            "Server=ISLEM;"
            "Database=job_finder;"
            "Trusted_Connection=yes;")
        self.connection_table = pyodbc.connect(cnxn_table)
        self.cursor_table = self.connection_table.cursor()

        table_name = 'Matching'
        #create_table_query = f"CREATE TABLE {table_name} (CV INT, JobOffer INT, JobTitle VARCHAR(255), SimilarityScore FLOAT)"
        #self.cursor_table.execute(create_table_query)
        #self.cursor_table.commit()
        results_df=pd.read_csv('accounts/ocr/results.csv')
        # Insert the results into the table
        for index, row in results_df.iterrows():
            cv = int(row['CV'])
            job_offer = int(row['Job_Offer'])
            job_title = row['job_title']
            similarity_score = float(row['Similarity_Score'])

            insert_query = f"INSERT INTO {table_name} (CV, JobOffer, JobTitle, SimilarityScore) VALUES (?, ?, ?, ?)"
            self.cursor_table.execute(insert_query, cv, job_offer, job_title, similarity_score)
            self.connection_table.commit()

    def close_connection(self):
        # Close the database connection
        self.cursor_table.close()
        self.connection_table.close()



