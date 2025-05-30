import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from dotenv import load_dotenv
from openai import OpenAI
import re
import numpy as np
# import comtypes.client
from pydantic import BaseModel, Field#type: ignore
from langchain.tools import StructuredTool

# Load OpenAI API key and Finnhub API key from environment variables
os.environ["OPENAI_API_KEY"] = "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"
os.environ["FINNHUB_API_KEY"] = "ct8o3k1r01qtkv5spi7gct8o3k1r01qtkv5spi80"  # Replace with your Finnhub API key

# Initialize the OpenAI client
client = OpenAI()

# def convert_docx_to_pdf(input_path: str, output_path: str) -> None:
#     """
#     Converts a .docx file to .pdf format using Microsoft Word via comtypes.

#     Args:
#         input_path (str): Path to the input .docx file.
#         output_path (str): Path to save the output .pdf file.

#     Returns:
#         None
#     """
#     try:
#         # Initialize Word application
#         word = comtypes.client.CreateObject("Word.Application")
#         word.Visible = False

#         # Open the Word document
#         doc = word.Documents.Open(input_path)

#         # Save as PDF
#         doc.SaveAs(output_path, FileFormat=17)  # 17 corresponds to PDF format in Word

#         # Close the document and quit Word
#         doc.Close()
#         word.Quit()

#         print(f"Conversion successful! PDF saved at: {output_path}")
#     except Exception as e:
#         print(f"An error occurred: {e}")



def chat_client(query: str) -> str:
    """
    A simple chat client that uses OpenAI's GPT model to answer a query.

    :param query: The user's question or prompt.
    :return: The AI's response as a string.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content.strip()