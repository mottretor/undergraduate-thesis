import mysql.connector
from mysql.connector import Error


class DatabaseUtil:

    def __init__(self):
        self.connection = None
        self.cursor = None

    def get_db_connection(self):
        if self.connection is None or not self.connection.is_connected():
            try:
                self.connection = mysql.connector.connect(host='localhost',
                                                          database='rms_using_sift',
                                                          user='rms_db_user',
                                                          password='pKeM_9#ryY'
                                                          ,use_pure=True)

            except Error as e:
                print("Database Error : ", e)

    def execute_query(self, query):
        if self.connection is None:
            self.get_db_connection()

        self.cursor = self.connection.cursor()
        print(query)
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        self.cursor.close()
        return rows

    def execute_insert(self, query):
        if self.connection is None:
            self.get_db_connection()

        self.cursor = self.connection.cursor()
        self.cursor.execute(query)
        self.connection.commit()
        rows = self.cursor.lastrowid
        self.cursor.close()
        return rows

    def execute_insert_blob(self, query, arg):
        if self.connection is None:
            self.get_db_connection()

        self.cursor = self.connection.cursor()
        self.cursor.execute(query, (arg,))
        self.connection.commit()
        rows = self.cursor.lastrowid
        self.cursor.close()
        return rows

    def execute_update_blob(self, query, arg):
        if self.connection is None:
            self.get_db_connection()

        self.cursor = self.connection.cursor()
        self.cursor.execute(query, (arg,))
        self.connection.commit()
        self.cursor.close()

    def execute_update(self, query):
        if self.connection is None:
            self.get_db_connection()

        self.cursor = self.connection.cursor()
        self.cursor.execute(query)
        self.connection.commit()
        self.cursor.close()

    @staticmethod
    def get_song_list():
        db_connection = None
        try:
            db_connection = mysql.connector.connect(host='localhost',
                                                    database='rmsdb',
                                                    user='rms_db_user',
                                                    password='pKeM_9#ryY')
        except Error as e:
            print("Database Error : ", e)
        if db_connection is not None:
            db_cursor = db_connection.cursor()
            db_cursor.execute("SELECT id, title FROM songs")
            rows = db_cursor.fetchall()
            db_cursor.close()
            db_connection.close()
            return rows
        return None

    def close(self):
        if self.connection is not None:
            self.connection.close()
        self.connection = None
