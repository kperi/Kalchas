# UI Project Documentation

## Overview
This project is a Next.js application that serves as a user interface for the functionality provided by the st_app Streamlit container. It includes features such as user authentication, PDF file uploading, cropping, and admin functionalities.

## Project Structure
The project is organized as follows:

- **app/**: Contains the main application pages and API routes.
  - **layout.tsx**: Defines the main layout of the application.
  - **page.tsx**: Entry point for the application.
  - **(auth)/login/page.tsx**: User authentication page.
  - **(main)/dashboard/page.tsx**: Dashboard displaying user-specific data.
  - **(main)/upload/page.tsx**: Page for uploading PDF files.
  - **(main)/admin/page.tsx**: Admin management page.
  - **api/user/route.ts**: API route for user-related operations.

- **components/**: Contains reusable components.
  - **AuthForm.tsx**: Component for user authentication input.
  - **FileUploader.tsx**: Component for managing file uploads and cropping.

- **lib/**: Contains utility functions.
  - **auth.ts**: Functions for authentication management.

- **public/**: Directory for static assets.

- **next.config.mjs**: Configuration settings for the Next.js application.

- **package.json**: npm configuration file listing dependencies and scripts.

- **tsconfig.json**: TypeScript configuration file.

## Features
- **Authentication**: Users can log in to access the application.
- **PDF Uploading**: Users can upload PDF files through the upload page.
- **Cropping Functionality**: Uploaded PDFs can be cropped as needed.
- **Admin Functionality**: Admin users can manage the application through the admin page.

## Setup Instructions
1. Clone the repository to your local machine.
2. Navigate to the `ui` directory.
3. Install the dependencies using npm:
   ```
   npm install
   ```
4. Run the development server:
   ```
   npm run dev
   ```
5. Open your browser and go to `http://localhost:3000` to access the application.

## Usage
- Navigate to the login page to authenticate.
- Once logged in, access the dashboard for user-specific functionalities.
- Use the upload page to upload PDF files.
- Admin users can access the admin page for management tasks.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.