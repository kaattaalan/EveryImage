import re

def refactor_database():
    with open('E:/Workspace/EveryImage/src/database.py', 'r') as f:
        code = f.read()

    # We want to replace basic conn = get_connection() ... conn.close()
    # It's better to just do this safely.
    # Actually, the user asked me to start fixing without undoing HDD optimisations.
    # I can just use AST or string replacement for the specific functions.
    pass

if __name__ == "__main__":
    refactor_database()
