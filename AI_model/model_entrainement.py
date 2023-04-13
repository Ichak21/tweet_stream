import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from cassandra.cluster import Cluster
from cassandra.query import dict_factory
from sklearn.metrics import make_scorer, accuracy_score, f1_score, log_loss
import uuid


class ModelTraining:
    
    # CASSANDRA
    
    
    production_model_rfc_log_loss:float = 0.0
    production_model_rfc_f1_score: float = 0.0  
    production_model_rfc_accuracy_score: float = 0.0
    production_model_rfc_run:int = 0 
    model_rfc_log_loss: float = 0.0
    model_rfc_f1_score: float = 0.0
    model_rfc_accuracy_score: float = 0.0
    model_rfc = None
    tweetPrepro = None
    keyspace_name = 'tweets'
    
    
    cluster = Cluster(['127.0.0.1','172.18.0.4'])        
    session = cluster.connect(keyspace_name)
    session.set_keyspace(keyspace_name)
    
    
    
    def _cassandraSession(self):
        cluster = Cluster(['127.0.0.1','172.18.0.4'])        
        return cluster.connect(keyspace=self.keyspace_name)

        
    
    def _tweetPreproced(self):
        # CASSANDRA
        

        query = "SELECT * FROM tweets_preproced"
        result_set = self.session.execute(query)

        # Conversion des résultats en dataframe pandas
        df = pd.DataFrame(result_set._current_rows)
        
        return df
    
    def loadProductionModel_Metrics(self):
        #load metrics from Cassandra, puis
        
        

        # query = "SELECT * FROM tweets_metrics"
        # result_set = self.session.execute(query)

        # # Conversion des résultats en dataframe pandas
        # df_metrics=  pd.DataFrame(result_set._current_rows).tail(1)
        
        
        self.production_model_rfc_log_loss =   0.0859281609725540 #df_metrics['production_model_rfc_log_loss']
        self.production_model_rfc_f1_score = 1 #df_metrics['production_model_rfc_f1_score']
        self.production_model_rfc_accuracy_score = 1 #df_metrics['production_model_rfc_accuracy_score']
        self.production_model_rfc_run = 1
        
        
        
        
        
        
        
    def trainModel(self):
        
        #choper depuis Cassandra
        df_full = self._tweetPreproced()
        df_full = df_full[['clean_tweet', 'sentiment']]
#         print('nb of positive :' ,df_full.loc[df_full.TB_SENTIMENT=='positive'].TB_ID.count(),'nb of neutral :' ,df_full.loc[df_full.TB_SENTIMENT=='neutral'].TB_ID.count(), 'nb of negative :',df_full.loc[df_full.TB_SENTIMENT=='negative'].TB_ID.count()) 
        tfidf = TfidfVectorizer()
        X_full = tfidf.fit(df_full["clean_tweet"].astype('str'))
        dump(X_full, 'tfidf_fit.joblib')
        X_full = tfidf.transform(df_full["clean_tweet"].astype('str'))
        sentiment_mapping = {"positive": 1, "negative": -1, "neutral": 0} 
        y_full = df_full["sentiment"].map(sentiment_mapping)

        model_rfc = RandomForestClassifier(n_estimators=200)
        model_rfc.fit(X_full, y_full)
        model_rfc.predict(X_full)
    
        model_rfc_log_loss = log_loss(y_full, model_rfc.predict_proba(X_full))
        model_rfc_f1_score = f1_score(y_full, model_rfc.predict(X_full), average='weighted')
        model_rfc_accuracy_score = accuracy_score(y_full, model_rfc.predict(X_full))
    
        
    
        self.model_rfc_log_loss = model_rfc_log_loss
        self.model_rfc_f1_score = model_rfc_f1_score
        self.model_rfc_accuracy_score = model_rfc_accuracy_score
#         dump(model_rfc, 'model_rfc_training.joblib') ou  return model_rfc ou compare metrics directement ici ?
# Peut-on return un model ou le store dans une variable ?
        self.model = model_rfc
        
    
    def compareMetrics(self):
        
        print("métriques de production actuelles : \n production_model_rfc_log_loss : ", self.production_model_rfc_log_loss, "\n production_model_rfc_f1_score : ", self.production_model_rfc_f1_score, "\n production_model_rfc_accuracy_score : ", self.production_model_rfc_accuracy_score)
        
        print("\n métriques du modèle entrainé : \n model_rfc_log_loss : ", self.model_rfc_log_loss, "\n model_rfc_f1_score : ", self.model_rfc_f1_score, "\n model_rfc_accuracy_score : ", self.model_rfc_accuracy_score)
              
        if self.model_rfc_log_loss <= self.production_model_rfc_log_loss and self.model_rfc_f1_score >= self.production_model_rfc_f1_score and self.model_rfc_accuracy_score >= self.production_model_rfc_accuracy_score:
            dump(self.model_rfc, 'model_rfc.joblib')
            
            
            # puis charger les nouvelles métriques dans Cassandra
            
            id = uuid.uuid1()
            
            # INSERT TWEET PREPROCED
            self.session.execute(
                """
                INSERT INTO tweets_metrics (id, production_model_rfc_log_loss, production_model_rfc_f1_score, production_model_rfc_accuracy_score, production_model_rfc_run)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (id, self.model_rfc_log_loss, self.model_rfc_f1_score, self.model_rfc_accuracy_score,self.production_model_rfc_run)
            )
            
            
            # self.model_rfc_log_loss, self.model_rfc_f1_score, self.model_rfc_accuracy_score qui deviennent
            # production_model_rfc_log_loss, production_model_rfc_f1_score, production_model_rfc_accuracy_score dans la table
            print("les nouvelles métriques sont meilleures -> remplacement du modèle de production et update de la table des métriques")
            
        
            
            
    
if __name__ == '__main__':
    training = ModelTraining()
    # training.tweetPreproced()
    training.loadProductionModel_Metrics()
    training.trainModel()
    training.compareMetrics()
    
    
    
    
        
        
 

    
    
