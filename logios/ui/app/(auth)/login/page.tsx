import React, { useState } from 'react';
import { useRouter } from 'next/router';
import AuthForm from '../../../components/AuthForm';
import { login } from '../../../lib/auth';

const LoginPage = () => {
    const [error, setError] = useState(null);
    const router = useRouter();

    const handleLogin = async (credentials) => {
        try {
            await login(credentials);
            router.push('/(main)/dashboard');
        } catch (err) {
            setError('Invalid username or password');
        }
    };

    return (
        <div>
            <h1>Login</h1>
            {error && <p style={{ color: 'red' }}>{error}</p>}
            <AuthForm onSubmit={handleLogin} />
        </div>
    );
};

export default LoginPage;