from app import create_app
import os

app = create_app()

if __name__ == '__main__':
    # Development server settings
    debug_mode = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5001))
    
    app.run(
        debug=debug_mode,
        host=host,
        port=port,
        use_reloader=debug_mode,  # Enable auto-reload in debug mode
        use_debugger=debug_mode,  # Enable debugger in debug mode
        threaded=True  # Handle multiple requests concurrently
    )