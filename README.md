# WebScraper-RAG
### Project Overview:
We built an AI - based Web Scraping + Q&A software that answers any of a user's questions about a website’s content. We leverage various fields of Web scraping, Large Language Models (LLMs), tokenization, embedding, vector databases and Retrieval Augmented Generation (RAG) and integrate them into our python project. 
### Working:
The user inputs a url for the website he/she wants to query about, along with any question that may have arised from browsing through it. We scrape the website’s html using textual tags. This data is tokenized, and embedded inside a vector database for retrieval. When the user asks any question, relevant information is retrieved from the vector space to generate an answer. This process is called Retrieval Augmented Generation (RAG). We use a Large Language Model locally to create a proper answer for the user's question which is displayed attractively on the output screen.
### Example Output:
![Screenshot 2024-11-20 164351](https://github.com/user-attachments/assets/9d294aca-88ca-4521-84d1-0f11843c2994)
We used this article from Times Of India and implemented our plan of action. After reading the article and asking a few questions, our model accurately answered these relevant questions - 
![Screenshot 2024-11-20 165823](https://github.com/user-attachments/assets/6854a027-2df1-4518-8be9-b101d5fab583)
![Screenshot 2024-11-20 164106](https://github.com/user-attachments/assets/165007b3-1679-499f-8df0-51daa9679d1c)
