# gradio_app.py

import gradio as gr
from ticket_predictor import predict_ticket

def process_ticket(text):
    result = predict_ticket(text)
    return (
        result['issue_type'],
        result['urgency_level'],
        result['entities']
    )

iface = gr.Interface(
    fn=process_ticket,
    inputs=gr.Textbox(lines=6, label="Enter Ticket Text"),
    outputs=[
        gr.Textbox(label="Predicted Issue Type"),
        gr.Textbox(label="Predicted Urgency Level"),
        gr.JSON(label="Extracted Entities")
    ],
    title="Ticket Classifier & Entity Extractor",
    description="Enter a customer support ticket. The app predicts issue type, urgency level, and extracts key entities."
)

if __name__ == "__main__":
    iface.launch()
