import streamlit as st
import json
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
import ast
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import os

if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None

st.title('Chatbot App For Conversation & Analysis Of Your Orders')
st.session_state.customer_id = st.text_input('Enter Your Customer Id')
if st.session_state.customer_id:
    st.text('Customer Id Entered Successfully')

def connection_to_db():
    username = 'postgres'
    password = '1234'
    host = 'localhost'
    port = '5432'
    dbname = 'postgres'

    engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{dbname}')
    conn = engine.connect() 
    return conn

def load_model():
    torch.random.manual_seed(0) 

    compute_dtype = torch.float16
    attn_implementation = 'sdpa'
    # quantization_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     bnb_8bit_compute_dtype=compute_dtype,
    #     bnb_8bit_use_double_quant=True,
    #     bnb_8bit_quant_type="nf8",
    # )
    adapter = "PavanDeepak/Peft_XLAM_toolcalling_db"
    model_name = "Salesforce/xLAM-1b-fc-r"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    #print(f"Starting to load the model {model_name} into memory")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation=attn_implementation,
    )
    model = PeftModel.from_pretrained(model, adapter)
    return model, tokenizer

if st.session_state.model is None or st.session_state.tokenizer is None:
    st.session_state.model, st.session_state.tokenizer = load_model()

def question(query, model, tokenizer):

    task_instruction = """
    You are an expert in composing functions. You are given a question and a set of possible functions. 
    Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
    ONLY GIVE OUT CORRECT FUNCTION CALLS DO NOT ADD ANY OTHER TEXT.
    If none of the functions can be used, point it out and refuse to answer. 
    If the given question lacks the parameters required by the function, also point it out.
    """.strip()

    format_instruction = """
    The output MUST strictly adhere to the following JSON format, and NO other text MUST be included.
    The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make tool_calls an empty list '[]'.
    ```
    {
        "tool_calls": [
        {"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},
        ... (more tool calls as required)
        ]
    }
    ```
    """.strip()


    get_customer_orders_api = {
    "name": "get_customer_orders",
    "description": "Retrieve 'N' prior orders for a customer using their customer ID",
    "parameters": {
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "The unique identifier for the customer"
            }
        },
        "required": ["limit"]
    }
}
    
    suggest_orders_api = {
        "name": "suggest_orders",
        "description": "Suggest additional orders based on a customer's previous purchases, USE ONLY WHEN CUSTOMERS ARE REQUESTING FOR RECOMMENDATIONS IN THE QUESTION",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The unique identifier for the customer"
                }
            },
        }
    }

    get_order_details_api = {
        "name": "get_order_details",
        "description": "Retrieve details of a specific order using the invoice number, USE ONLY WHEN INVOICE IS MENTIONED IN THE QUESTION",
        "parameters": {
            "type": "object",
            "properties": {
                "invoice_no": {
                    "type": "string",
                    "description": "The invoice number of the order"
                }
            },
            "required": ["invoice_no"]
        }
    }

    get_order_history_by_items_api = {
        "name": "get_order_history_by_items",
        "description": "Retrieve all orders of a specific items for a customer, USE ONLY WHEN ITEMS IS MENTIONED IN THE QUESTION",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The unique identifier for the customer"
                },
                "items": {
                    "type": "string",
                    "description": "The particular item purchased by the customer"
                }
            },
            "required": ["items"]
        }
    }

    get_frequent_purchases_api = {
        "name": "get_frequent_purchases",
        "description": "Get frequently purchased items by a customer",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The unique identifier for the customer"
                }
            },
        }
    }

    calculate_total_spend_api = {
        "name": "calculate_total_spend",
        "description": "Calculate the total amount spent by a customer.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The unique identifier for the customer"
                }
            },
        }
    }

    get_customer_profile_api = {
        "name": "get_customer_profile",
        "description": "Retrieve the profile information of a customer",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The unique identifier for the customer"
                }
            },
        }
    }

    openai_format_tools = [get_customer_orders_api , suggest_orders_api, get_order_details_api, get_order_history_by_items_api, 
                           get_frequent_purchases_api, calculate_total_spend_api, get_customer_profile_api]

    def convert_to_xlam_tool(tools):
        ''''''
        if isinstance(tools, dict):
            return {
                "name": tools["name"],
                "description": tools["description"],
                "parameters": {k: v for k, v in tools["parameters"].get("properties", {}).items()}
            }
        elif isinstance(tools, list):
            return [convert_to_xlam_tool(tool) for tool in tools]
        else:
            return tools

    def build_prompt(task_instruction: str, format_instruction: str, tools: list, query: str):
        prompt = f"[BEGIN OF TASK INSTRUCTION]\n{task_instruction}\n[END OF TASK INSTRUCTION]\n\n"
        prompt += f"[BEGIN OF AVAILABLE TOOLS]\n{json.dumps(xlam_format_tools)}\n[END OF AVAILABLE TOOLS]\n\n"
        prompt += f"[BEGIN OF FORMAT INSTRUCTION]\n{format_instruction}\n[END OF FORMAT INSTRUCTION]\n\n"
        prompt += f"[BEGIN OF QUERY]\n{query}\n[END OF QUERY]\n\n"
        return prompt
        
    xlam_format_tools = convert_to_xlam_tool(openai_format_tools)
    content = build_prompt(task_instruction, format_instruction, xlam_format_tools, query)

    messages=[
        { 'role': 'user', 'content': content}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    outputs = model.generate(inputs, max_new_tokens=512, temperature=1, do_sample=True, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

def get_customer_orders_def(customer_id='C3628', limit=5, **kwargs):
    data =  []
    conn = connection_to_db()
    query = f"SELECT * FROM customers_shopping_data WHERE customer_id = '{customer_id}' ORDER BY order_date DESC LIMIT '{limit}'"
    result = conn.execute(text(query))
    conn.close()
    for row in result:
        data.append(row)
    return data

def suggest_orders_def(customer_id='C3628', limit=5, similarity_threshold=0.8, **kwargs):
    conn = connection_to_db()
    query = """
    SELECT customer_id, items, SUM(quantity) as total_quantity
    FROM customers_shopping_data
    GROUP BY customer_id, items
    """
    
    purchase_history = pd.read_sql_query(query, conn)
    conn.close()

    user_item_matrix = purchase_history.pivot(index='customer_id', columns='items', values='total_quantity').fillna(0)

    if customer_id not in user_item_matrix.index:
        return []

    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(user_item_matrix)

    customer_idx = user_item_matrix.index.get_loc(customer_id)
    distances, indices = model.kneighbors(user_item_matrix.iloc[customer_idx, :].values.reshape(1, -1), n_neighbors=len(user_item_matrix))

    similar_customers = [user_item_matrix.index[i] for i, dist in zip(indices.flatten(), distances.flatten()) if dist <= 1 - similarity_threshold and user_item_matrix.index[i] != customer_id]

    if len(similar_customers) < 10:
        similar_customers = user_item_matrix.index[indices.flatten()[1:11]]

    similar_customers_purchases = user_item_matrix.loc[similar_customers].sum(axis=0).sort_values(ascending=False)
    
    target_customer_purchases = user_item_matrix.loc[customer_id]
    target_categories = target_customer_purchases[target_customer_purchases > 0].index
    recommended_products = similar_customers_purchases[~similar_customers_purchases.index.isin(target_categories)]
    
    return recommended_products.head(limit).index.tolist()

def get_order_details_def(invoice_no, **kwargs):
    data =  []
    conn = connection_to_db()
    query = f"SELECT * FROM customers_shopping_data WHERE invoice_no = '{invoice_no}'"
    result = conn.execute(text(query))
    conn.close()
    for row in result:
        data.append(row)
    return data

def get_order_history_by_items_def(customer_id='C3628', **kwargs):
    data =  []
    conn = connection_to_db()
    query = f"SELECT items, SUM(quantity) as total_quantity FROM customers_shopping_data WHERE customer_id = '{customer_id}' GROUP BY items ORDER BY total_quantity DESC"
    result = conn.execute(text(query))
    conn.close()
    for row in result:
        data.append(row)
    return data

def get_frequent_purchases_def(customer_id='C3628', **kwargs):
    data =  []
    conn = connection_to_db()
    query = f"SELECT items, SUM(quantity) as total_quantity FROM customers_shopping_data WHERE customer_id = '{customer_id}' GROUP BY items ORDER BY total_quantity DESC"
    result = conn.execute(text(query))
    conn.close()
    for row in result:
        data.append(row)
    return data

def calculate_total_spend_def(customer_id='C3628', **kwargs):
    data = []
    conn = connection_to_db()
    query = f"SELECT SUM(price) FROM customers_shopping_data WHERE customer_id = '{customer_id}'"
    result = conn.execute(text(query))
    conn.close()
    for row in result:
        data.append(row)
    return data

def get_top_categories_def(customer_id='C3628', **kwargs):
    data =  []
    conn = connection_to_db()
    query = f"SELECT items, SUM(quantity) as total_quantity FROM customers_shopping_data WHERE customer_id = '{customer_id}' GROUP BY items ORDER BY total_quantity DESC LIMIT 5"
    result = conn.execute(text(query))
    conn.close()
    for row in result:
        data.append(row)
    return data

def get_customer_profile_def(customer_id='C3628', **kwargs):
    data =  []
    conn = connection_to_db()
    query = f"SELECT * FROM customers_shopping_data WHERE customer_id = '{customer_id}'"
    result = conn.execute(text(query))
    conn.close()
    for row in result:
        data.append(row)
    return data

functions_list = {
    "get_customer_orders": get_customer_orders_def,
    "suggest_orders": suggest_orders_def,
    "get_order_details": get_order_details_def,
    "get_order_history_by_items": get_order_history_by_items_def,
    "get_frequent_purchases": get_frequent_purchases_def,
    "calculate_total_spend": calculate_total_spend_def,
    "get_top_categories": get_top_categories_def,
    "get_customer_profile": get_customer_profile_def
}

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], str):
            st.markdown(message["content"])
        else:
            st.plotly_chart(message["content"])

if prompt := st.chat_input("Please enter your questions:"):
    query = f"for customer_id: {st.session_state.customer_id} {prompt} for customer_id: {st.session_state.customer_id}"
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner('Processing your request...'):
            final_prompt = f"user_query:{query}"
            out = question(final_prompt, st.session_state.model, st.session_state.tokenizer)
            output_cleaned = ast.literal_eval(out)
            try:
                data = []
                for tool in output_cleaned['tool_calls']:
                    result = functions_list[tool['name']](**tool['arguments'])
                    data.append(result)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                data = output_cleaned

            response_content = f"I've processed your request. Here's a summary of the actions taken:\n\n"
            tool_call_dict = {}
            for tool in output_cleaned['tool_calls']:
                if tool['name'] not in tool_call_dict:
                    tool_call_dict[tool['name']] = 1
                    response_content += f"- Used the {tool['name']} function with arguments: {tool['arguments']}\n"
                else:
                    pass
            response_content += f"\nBased on these actions, here's the final result:\n{data}"
            st.markdown(response_content)
            st.session_state.messages.append({"role": "assistant", "content": response_content})

if st.button("Reset", type="primary"):
    st.session_state.messages = []
    st.session_state.customer_id = []
    def clear():
        os.system('cls' if os.name == 'nt' else 'clear')
    clear()
    st.rerun()
