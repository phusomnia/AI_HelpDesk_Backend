from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import os
import jwt
from src.SharedKernel.utils.yamlenv import load_env_yaml

config = load_env_yaml()

class JWTProvider:
    def __init__(self):
        self.secret_key = config.jwt.secret
        self.algorithm = "HS256"
        self.expire_minutes = 60
    
    def create_access_token(self, data: Dict[str, Any]):
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(minutes=self.expire_minutes)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str):
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def is_token_expired(self, token: str) -> bool:
        payload = self.verify_token(token)
        if payload is None:
            return True
        
        exp = payload.get("exp")
        if exp is None:
            return True
        
        return datetime.now(timezone.utc).timestamp() > exp