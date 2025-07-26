\# MOSDAC AI Chatbot: Your Satellite Data Assistant 🛰️



A Streamlit-powered AI chatbot designed to provide instant information about the Meteorological \& Oceanographic Satellite Data Archival Centre (MOSDAC). Get quick answers on ISRO's satellite missions, various data products, and how to access earth observation data.



\## ✨ Features

\* \*\*Intelligent Q\&A:\*\* Ask questions about MOSDAC, specific satellites (e.g., INSAT-3DR, Oceansat-2, RISAT-1, Megha-Tropiques), and data types.

\* \*\*Data Product Information:\*\* Learn about Level-1, Level-2, meteorological, oceanographic, and other satellite data products.

\* \*\*Easy Access:\*\* Get guidance on how to register and download data from the MOSDAC portal.

\* \*\*Built with:\*\* Google Gemini, LangChain, ChromaDB, and Streamlit.



\## 🚀 Try the Live Demo

\*\*\[Click here to chat with the MOSDAC AI Assistant!](YOUR\_STREAMLIT\_APP\_URL\_HERE)\*\*

\*(Replace `YOUR\_STREAMLIT\_APP\_URL\_HERE` with the actual public URL of your deployed Streamlit app, like `https://mosdac-info-bot.streamlit.app/`)\*



\## 🛠️ How it Works

This chatbot utilizes a Retrieval Augmented Generation (RAG) approach:

1\.  \*\*Data Collection:\*\* Information is scraped from the MOSDAC website (or provided via `mosdac\_data.json`).

2\.  \*\*Vector Database:\*\* The collected data is embedded using Google Generative AI Embeddings and stored in a ChromaDB vector store.

3\.  \*\*Querying:\*\* When you ask a question, relevant information is retrieved from the vector database.

4\.  \*\*LLM Generation:\*\* The retrieved information is sent to a large language model (Google Gemini 1.5 Flash) to generate a coherent and accurate answer.



\## 💻 Local Development Setup (for Developers)

To run this chatbot on your local machine:



1\.  \*\*Clone the repository:\*\*

&nbsp;   ```bash

&nbsp;   git clone \[https://github.com/29082006krishna/mosdac-chatbot-app.git](https://github.com/29082006krishna/mosdac-chatbot-app.git)

&nbsp;   cd mosdac-chatbot-app

&nbsp;   ```

2\.  \*\*Create and activate a virtual environment:\*\*

&nbsp;   ```bash

&nbsp;   python -m venv venv

&nbsp;   .\\venv\\Scripts\\activate # On Windows

&nbsp;   # source venv/bin/activate # On macOS/Linux

&nbsp;   ```

3\.  \*\*Install dependencies:\*\*

&nbsp;   ```bash

&nbsp;   pip install -r requirements.txt

&nbsp;   ```

4\.  \*\*Set your Google API Key:\*\*

&nbsp;   Create a `.env` file in the root directory of the project and add your API key:

&nbsp;   ```

&nbsp;   GOOGLE\_API\_KEY="YOUR\_GOOGLE\_API\_KEY\_HERE"

&nbsp;   ```

&nbsp;   (Get your API key from \[Google AI Studio](https://aistudio.google.com/app/apikey))

5\.  \*\*Build the vector database:\*\*

&nbsp;   ```bash

&nbsp;   python create\_vector\_db.py

&nbsp;   ```

6\.  \*\*Run the Streamlit app:\*\*

&nbsp;   ```bash

&nbsp;   streamlit run chatbot\_app.py

&nbsp;   ```



\## 📄 License

This project is open-source and available under the \[MIT License](https://opensource.org/licenses/MIT).

