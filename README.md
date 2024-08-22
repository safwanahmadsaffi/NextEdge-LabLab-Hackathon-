# Project README

## Overview

This project consists of a series of API tools designed to interact with a customer order database. Each tool serves a distinct purpose, providing functionalities such as retrieving customer orders, suggesting additional purchases, and calculating total spend. These tools are built to streamline customer interactions by enabling efficient data retrieval and analysis based on specific customer requests.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [API Tools](#api-tools)
  - [Get Customer Orders](#get-customer-orders)
  - [Suggest Orders](#suggest-orders)
  - [Get Order Details](#get-order-details)
  - [Get Order History by Items](#get-order-history-by-items)
  - [Get Frequent Purchases](#get-frequent-purchases)
  - [Calculate Total Spend](#calculate-total-spend)
  - [Get Customer Profile](#get-customer-profile)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/payne19/Lab_Lab_AI_Chatbot.git
   ```

2. Navigate to the project directory:

   ```bash
   cd your-installed-directory
   ```

3. Install the necessary dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the ui_chatbot.py file:
   ```bash
   streamlit run ui_chatbot.py
   ```

## Usage

Each tool is designed to perform a specific query against the customer order database. These tools can be integrated into a larger system, such as a chatbot or customer support platform, to provide real-time data retrieval and suggestions based on customer interactions.

## API Tools

### Get Customer Orders

**Name:** `get_customer_orders`

**Description:** Retrieve 'N' prior orders for a customer using their customer ID.

**Parameters:**
- `customer_id`: The unique identifier for the customer.
- `required`: `["limit"]`

### Suggest Orders

**Name:** `suggest_orders`

**Description:** Suggest additional orders based on a customer's previous purchases. Use this API only when customers request recommendations in their queries.

**Parameters:**
- `customer_id`: The unique identifier for the customer.

### Get Order Details

**Name:** `get_order_details`

**Description:** Retrieve details of a specific order using the invoice number. Use this API only when the invoice is mentioned in the query.

**Parameters:**
- `invoice_no`: The invoice number of the order.
- `required`: `["invoice_no"]`

### Get Order History by Items

**Name:** `get_order_history_by_items`

**Description:** Retrieve all orders of specific items for a customer. Use this API only when items are mentioned in the query.

**Parameters:**
- `customer_id`: The unique identifier for the customer.
- `items`: The particular item purchased by the customer.
- `required`: `["items"]`

### Get Frequent Purchases

**Name:** `get_frequent_purchases`

**Description:** Get frequently purchased items by a customer.

**Parameters:**
- `customer_id`: The unique identifier for the customer.

### Calculate Total Spend

**Name:** `calculate_total_spend`

**Description:** Calculate the total amount spent by a customer.

**Parameters:**
- `customer_id`: The unique identifier for the customer.

### Get Customer Profile

**Name:** `get_customer_profile`

**Description:** Retrieve the profile information of a customer.

**Parameters:**
- `customer_id`: The unique identifier for the customer.

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and create a pull request.

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License

This project is licensed under the MIT License.
