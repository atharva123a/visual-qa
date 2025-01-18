import redis

# Initialize a global variable for the Redis client
redis_client = None

def initialize_redis_connection(host='localhost', port=6379, db=0, password=None) -> redis.StrictRedis:
    global redis_client
    if redis_client is None:
        try:
            # Create a Redis client
            redis_client = redis.StrictRedis(host=host, port=port, db=db, password=password, decode_responses=True)
            
            # Test the connection
            redis_client.ping()
            print(f"Connected to Redis server at {host}:{port}, database {db}.")
        
        except redis.ConnectionError as e:
            print(f"Failed to connect to Redis server: {e}")
            raise
    return redis_client

def return_redis_client():
    return redis_client

redis_client = initialize_redis_connection()