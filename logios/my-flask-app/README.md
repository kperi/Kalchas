# My Flask App

This project is a web application built with Flask that allows users to upload images and PDF files, preview images, and crop them. It includes user authentication and maintains a structured layout similar to a Streamlit app.

## Features

- User authentication (login/logout)
- Image and PDF file uploads (up to 100MB)
- Image preview before cropping
- Cropping functionality for uploaded images
- Responsive layout with CSS styling
- JavaScript for client-side interactions

## Project Structure

```
my-flask-app
├── app
│   ├── __init__.py          # Initializes the Flask application
│   ├── routes.py            # Defines application routes
│   ├── auth.py              # Manages user authentication
│   ├── forms.py             # Contains form classes for authentication and uploads
│   ├── utils.py             # Utility functions for image processing
│   ├── static
│   │   ├── css              # CSS files for styling
│   │   └── js               # JavaScript files for client-side functionality
│   └── templates
│       ├── layout.html      # Base template for the application
│       ├── index.html       # Main page for uploads
│       ├── login.html       # Login page template
│       ├── image_preview.html# Template for image preview
│       └── crop_menu.html   # Template for cropping interface
├── uploads                   # Directory for uploaded files
├── requirements.txt          # Project dependencies
├── config.py                # Configuration settings
├── run.py                   # Entry point to run the application
└── README.md                # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd my-flask-app
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Configure the application by editing `config.py` as needed.

## Usage

1. Run the application:
   ```
   python run.py
   ```

2. Open your web browser and go to `http://127.0.0.1:5000`.

3. Use the login page to authenticate, then upload images or PDFs on the main page.

4. Preview and crop images as needed.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.