from fastapi import Depends
from starlette import status
from src.Domain.base_entities import Accounts
from src.Features.AuthAPI.AccountDTO import CreateAccountRequest, LoginAccountRequest, SearchAccountRequest, UpdateAccountRequest
from src.Features.AuthAPI.AccountRepository import UserRepository
import bcrypt
from src.Features.AuthAPI.JWTProvider import JWTProvider
from src.SharedKernel.exception.APIException import APIException

class AuthService:
    def __init__(
        self, 
        repo: UserRepository = Depends(), 
        jwt_provider: JWTProvider = Depends()
    ):
        self.repo = repo
        self.jwt_provider = jwt_provider
        
    async def search_accounts(self, req: SearchAccountRequest):
        return await self.repo.search_accounts(req)

    async def register_account(self, dto: CreateAccountRequest):
        existed_account = await self.repo.find_by_username(dto.username)
        if existed_account:
            raise APIException(
                "Username already exists",
                status_code=status.HTTP_400_BAD_REQUEST
            )

        account = dto.to_entity()
        
        return await self.repo.save(account)
    
    # 
    async def login_account(self, dto: LoginAccountRequest):
        account = await self.repo.find_by_username(dto.username)
        if not account:
            raise APIException(
                "Invalid email or password",
                status_code=status.HTTP_400_BAD_REQUEST
            )

        is_valid = bcrypt.checkpw(dto.password.encode('utf-8'), account['password'].encode('utf-8'))
        if not is_valid:
            raise APIException(
                "Invalid email or password",
                status_code=status.HTTP_400_BAD_REQUEST
            )

        user_data = {
            "username": account['username'],
            "role": account['role'],
            "user_id": str(account['id'])
        }

        access_token = self.jwt_provider.create_access_token(user_data)

        return access_token

    # 
    async def edit_account(self, id: str, dto: UpdateAccountRequest):
        selected_account = await self.repo.find_by_id(id)
        if not selected_account:
            raise APIException(
                "User not found",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        account = dto.to_entity(selected_account)
        
        return await self.repo.update(account)\

    async def delete_account(self, id: str):
        user = await self.repo.find_by_id(id)
        if not user:
            raise APIException(
                "User not found",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        return await self.repo.delete(user)

    def verify_password(self, plain_password, hashed_password):
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))