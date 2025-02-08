from flask import Flask,render_template,redirect,url_for,request
app = Flask(__name__)
import joblib
import numpy as np
import bcrypt
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd 


import mysql.connector
mydb = mysql.connector.connect(
    host='localhost',
    port=3306,          
    user='root',        
    passwd='',          
    database='rumor'  
)

mycur = mydb.cursor()




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/registration', methods=['POST', 'GET'])
def registration():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phonenumber = request.form['phonenumber']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']
        age = request.form['age']
        place = request.form['place']

        if password == confirmpassword:
            # Check if user already exists
            sql = 'SELECT * FROM users WHERE email = %s'
            val = (email,)
            mycur.execute(sql, val)
            data = mycur.fetchone()
            
            if data is not None:
                msg = 'User already registered!'
                return render_template('registration.html', msg=msg)
            else:
                # Hash the password
                hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                
                # Insert new user
                sql = 'INSERT INTO users (name, email, phonenumber, password, age, place) VALUES (%s, %s, %s, %s, %s, %s)'
                val = (name, email, phonenumber, hashed_password.decode('utf-8'), age, place)
                mycur.execute(sql, val)
                mydb.commit()
                
                msg = 'User registered successfully!'
                return render_template('login.html', msg=msg)
        else:
            msg = 'Passwords do not match!'
            return render_template('registration.html', msg=msg)
    
    return render_template('registration.html')


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        sql = 'SELECT * FROM users WHERE email=%s'
        val = (email,)
        mycur.execute(sql, val)
        data = mycur.fetchone()

        if data:
            stored_password_hash = data[3] 
            if bcrypt.checkpw(password.encode('utf-8'), stored_password_hash.encode('utf-8')):
                return render_template('prediction.html')
            else:
                msg = 'Password does not match!'
                return render_template('login.html', msg=msg)
        else:
            msg = 'User with this email does not exist. Please register.'
            return render_template('login.html', msg=msg)
    
    return render_template('login.html')
    


@app.route('/prediction',methods=['POST','GET'])
def prediction():
    if request.method == 'POST':
        import torch
        import numpy as np
        
        def load_model(model_path, model_class, tokenizer_class):
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Initialize the model and tokenizer
            model = model_class.from_pretrained("bert-base-uncased" if "bert" in model_path else "roberta-base", 
                                                num_labels=2, 
                                                output_attentions=False,
                                                output_hidden_states=False,
                                                ignore_mismatched_sizes=True)  # Add this to ignore size mismatches

            tokenizer = tokenizer_class.from_pretrained("bert-base-uncased" if "bert" in model_path else "roberta-base")
            
            # Load the trained model parameters, ignoring missing keys
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
            model.eval()  # Set the model to evaluation mode
            
            return model, tokenizer


        def predict(input_text, model, tokenizer):
            # Encode the input text
            encoded_dict = tokenizer.encode_plus(
                                text=input_text,
                                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                max_length=64,  # Pad & truncate all sentences.
                                pad_to_max_length=True,
                                return_attention_mask=True,
                                return_tensors='pt',  # Return PyTorch tensors
                        )
            
            # Extract inputs from the encoded dictionary
            input_ids = encoded_dict['input_ids']
            attention_mask = encoded_dict['attention_mask']
            
            # Model prediction
            with torch.no_grad():
                outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
            
            logits = outputs[0]
            predicted_prob = torch.softmax(logits, dim=1).numpy().flatten()  # Convert logits to probabilities
            prediction = torch.argmax(logits, dim=1).numpy().flatten()[0]  # Get the predicted class index
            
            return prediction, predicted_prob

        # Load the model and tokenizer
        model_path = 'bert.bin'  # or 'roberta_model.bin'
        model_class = BertForSequenceClassification  # or RobertaForSequenceClassification for RoBERTa
        tokenizer_class = BertTokenizer  # or RobertaTokenizer for RoBERTa

        model, tokenizer = load_model(model_path, model_class, tokenizer_class)

        # Predict user input

        user_input = request.form['text']
        print(user_input)

        prediction, probabilities = predict(user_input, model, tokenizer) 
        if prediction == 1:
            msg = 'NOT RUMOR'
            suggestion = (
                'Great news! The information appears to be reliable. '
                'Continue with your plans confidently, and remember, staying informed through trusted sources always empowers you. Keep up the good work!'
            )
            return render_template('prediction.html', msg=msg, suggestion=suggestion)
        else:
            msg = 'RUMOR'
            suggestion = (
                'Heads up! This information might be misleading. '
                'Use this as an opportunity to verify and deepen your understanding. '
                'By critically evaluating information, you contribute to a more informed and trustworthy community. Stay curious and discerning!'
            )

            return render_template('prediction.html', msg=msg, suggestion=suggestion)
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug = True)