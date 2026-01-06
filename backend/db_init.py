def init_db(conn):
    cursor = conn.cursor()
    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) UNIQUE,
            password VARCHAR(255)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            name VARCHAR(255),
            description TEXT,
            persona TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_projects_user
                FOREIGN KEY (user_id) REFERENCES users(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INT AUTO_INCREMENT PRIMARY KEY,
            project_id INT NOT NULL,
            title VARCHAR(255),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            lora_id INT,
            CONSTRAINT fk_chats_project
                FOREIGN KEY (project_id) REFERENCES projects(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INT AUTO_INCREMENT PRIMARY KEY,
            chat_id INT NOT NULL,
            role VARCHAR(50),
            content TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_messages_chat
                FOREIGN KEY (chat_id) REFERENCES chats(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INT AUTO_INCREMENT PRIMARY KEY,
            project_id INT NOT NULL,
            filename VARCHAR(255),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uniq_project_filename (project_id, filename),
            CONSTRAINT fk_documents_project
                FOREIGN KEY (project_id) REFERENCES projects(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            document_id INT NOT NULL,
            chunk TEXT,
            embedding JSON,
            CONSTRAINT fk_chunks_document
                FOREIGN KEY (document_id) REFERENCES documents(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_files (
            id INT AUTO_INCREMENT PRIMARY KEY,
            chat_id INT NOT NULL,
            file_id INT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_chatfiles_chat
                FOREIGN KEY (chat_id) REFERENCES chats(id)
                ON DELETE CASCADE,
            CONSTRAINT fk_chatfiles_document
                FOREIGN KEY (file_id) REFERENCES documents(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

        """)


        ## 로라 
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS lora_profiles (
            id INT AUTO_INCREMENT PRIMARY KEY,
            project_id INT NOT NULL,
            name VARCHAR(255),
            description TEXT,
            purpose TEXT,
            status VARCHAR(50),
            adapter_path TEXT,
            base_model VARCHAR(255),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_lora_project
                FOREIGN KEY (project_id) REFERENCES projects(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

        """)
        conn.commit()
    finally:
        cursor.close()