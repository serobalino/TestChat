import json
import os
from huggingface_hub import HfApi, login
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset

login(os.getenv('TOKEN'))

# Cargar datos desde JSON
with open('data.json', 'r') as file:
    data = json.load(file)

# Preprocesar datos
texts = [thesis['RESUMEN'] for thesis in data['theses']]
labels = [0] * len(texts)  # Ajusta las etiquetas según sea necesario

# Convertir datos a Dataset
dataset = Dataset.from_dict({'text': texts, 'label': labels})
dataset = dataset.train_test_split(test_size=0.2)

# Cargar modelo y tokenizer
model_name = os.getenv('MODEL_NAME', 'distilbert-base-uncased')
my_model_name = os.getenv('MY_MODEL')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Configuración de entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],  # Proporcionar eval_dataset
)

trainer.train()

# Guardar el modelo
model.save_pretrained(my_model_name)
tokenizer.save_pretrained(my_model_name)

# Subir el modelo a Hugging Face
path_model = "/app/" + my_model_name
print("Modelo: ",path_model)
api = HfApi()
api.upload_folder(
    folder_path=path_model,
    path_in_repo=".",
    repo_id=my_model_name,
    token=os.getenv('TOKEN')
)
