import React from 'react';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import { getUserData } from '../../api/user/route';
import FileUploader from '../../../components/FileUploader';

const Dashboard = () => {
    const [userData, setUserData] = useState(null);
    const router = useRouter();

    useEffect(() => {
        const fetchUserData = async () => {
            try {
                const data = await getUserData();
                setUserData(data);
            } catch (error) {
                console.error('Error fetching user data:', error);
                router.push('/auth/login');
            }
        };

        fetchUserData();
    }, [router]);

    if (!userData) {
        return <div>Loading...</div>;
    }

    return (
        <div>
            <h1>Welcome to your Dashboard, {userData.name}</h1>
            <FileUploader />
            {/* Additional dashboard functionalities can be added here */}
        </div>
    );
};

export default Dashboard;