

##### **🏛️ Civic Insight Analyzer**



AI-powered civic transparency app that connects Portland residents' feedback with government council actions using semantic search and NLP, Deployed live on Streamlit Cloud.



🔗 https://civic-insight-app-vdnjd2sa7wmaddtre2tgyk.streamlit.app





##### **What It Does**



The Civic Insight Analyzer makes Portland's public civic data searchable and understandable. It connects two datasets — citizen feedback and city council orders — and lets users explore them through natural language search and side-by-side comparison.







##### **Features**



💬 Citizen Feedback Search

Type any question in plain English and the app retrieves the most relevant feedback entries using semantic similarity. Returns a plain-language summary of what residents are concerned about.



🏛️ Council Orders Search

Search through 389 council orders by topic. Returns matching orders with category, date, and a summary of government actions taken.



⚖️ Compare \& Connect

Select an issue category and neighborhood to see community concerns alongside government responses side by side, revealing where civic voices and policy decisions align or diverge.







##### **Tech Stack**



| Layer               | Tools 

|------------------   |--------------------------------------------       

| Frontend            | Streamlit                                  

| Semantic Search     | Sentence Transformers (`all-MiniLM-L6-v2`) 

| Similarity Matching | Scikit-learn (Cosine Similarity) 

| Data Processing     | Pandas 

| Deployment          | Streamlit Community Cloud 







##### **Dataset**



Source: City of Portland 2023 — Civic Feedback \& Council Orders



| Dataset          | Records 

|----------------  |-------------

| Citizen Feedback | 500 entries 

| Council Orders   | 389 orders 

| Issue Categories | 7 

| Neighborhoods    | 10 







##### **How to Run Locally**



bash

\# Clone the repo

git clone https://github.com/jaiswal-arch/civic-insight-app



\# Create and activate environment

conda create -n civic\_app python=3.11 -y

conda activate civic\_app



\# Install dependencies

pip install -r requirements.txt



\# Run the app

streamlit run app.py









##### **Project Structure**



civic-insight-app/

├── app.py                        # Main Streamlit application

├── requirements.txt              # Dependencies

├── .python-version               # Python version pin (3.11)

└── City of Portland 2023.xlsx   # Source data



=======
🔗 Live App: https://civic-insight-app-vdnjd2sa7wmaddtre2tgyk.streamlit.app
>>>>>>> fc65a7e2cffae4c4fccd58263da052fc7a5693ba
