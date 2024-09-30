import pandas as pd
import numpy as np
import warnings
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

class Recommendation:
    def __init__(self, data_path=['./Groceries_data_train.csv','./Groceries data test.csv'], similarity_matrix_path=None):
        self.df = pd.concat([pd.read_csv(file_path) for file_path in data_path])
        self.df_item = self.df['itemDescription']
        self.df_number = sorted(self.df.Member_number.unique().tolist())
        self.group = self.df.groupby('Member_number')['itemDescription'].agg(lambda x: x.tolist()).to_frame()
        self.dictionary = self.df.groupby('itemDescription').apply(lambda dfg: dfg.shape[0]).to_dict()
        self.num_dict = {number: index for index, number in enumerate(self.df_number)}
        self.item_dict = dict(zip(self.df_item.unique(), np.arange(len(self.df_number) - 1)))
        self.purchase_matrix = self._create_purchase_matrix()
        if similarity_matrix_path:
            self.similarity_matrix = self._load_similarity_matrix(similarity_matrix_path)
        else:
            self.similarity_matrix = self._calculate_similarity_matrix()
    
    def _create_purchase_matrix(self):
        df_utility = pd.DataFrame(columns=self.df_item.unique())
        for number in self.df_number:
            row += [1 if item in self.group.itemDescription[number] else 0 for item in df_utility.columns]
            df_utility.loc[number] = row
        return csr_matrix(df_utility.values)
    
    def _calculate_similarity_matrix(self):
        num_customers = self.purchase_matrix.shape[0]
        similarity_matrix = np.zeros((num_customers, num_customers))

        for i in range(num_customers):
            purchase_vector_i = self.purchase_matrix[i]
            similarities = cosine_similarity(purchase_vector_i, self.purchase_matrix)
            similarity_matrix[i] = similarities[0]
        return csr_matrix(similarity_matrix)
    
    def get_recommendation(self, customer_number, k,method_number):
        a = self.num_dict.get(customer_number)
        similarities = self.similarity_matrix[a].toarray().flatten()
        sort_in = np.argsort(similarities)[::-1]
        neighbors = sort_in[1:k+1]
        
        recommendation_set = set()
        for items in neighbors:
            items_list = self.group.iloc[items].values.tolist()
            for sublist in items_list:
                recommendation_set.update(sublist)
        
        recommendations = list(recommendation_set)
        frequency = [self.dictionary.get(item) for item in recommendations]
        
        sorted_recommendations = [x for _, x in sorted(zip(frequency, recommendations), reverse=True)][:k]
        if method_number==1:
            return sorted_recommendations
        else:
            pattern=[]
            pattern['antecedents']=pattern['antecedents'].str.replace(r'[\[\],]', '').str.replace("'","")
            pattern['consequents']=pattern['consequents'].str.replace(r'[\[\],]', '').str.replace("'", "")
            pat_arr=pattern[['antecedents','consequents']].values
            pat_dict={row[0]: row[1] for row in pat_arr[0:]}
            # Recommending more Items based on patterns
            for j in sorted_recommendations:
                comp=pat_dict.get(j)
                if ((comp!=None) and (comp not in sorted_recommendations)):
                        sorted_recommendations.append(pat_dict.get(j))