import React from 'react';
import { AuthProvider } from '../lib/auth';
import Layout from './layout';

const HomePage = () => {
    return (
        <AuthProvider>
            <Layout>
                <h1>Welcome to the Application</h1>
                <p>Please navigate to the login page to access your account.</p>
            </Layout>
        </AuthProvider>
    );
};

export default HomePage;