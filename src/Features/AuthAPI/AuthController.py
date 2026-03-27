# from fastapi import APIRouter, Depends
# from starlette import status
# from src.Features.AuthAPI.AccountDTO import AccountCreateDTO, AccountSearchRequest, AccountUpdateDTO, AccountLoginDTO
# from src.Features.AuthAPI.AuthService import AuthService
# from src.Shared.base.APIResponse import APIResponse

# service = AuthService()

# router = APIRouter(
#     prefix="/api/v1/auth",
#     tags=["Auth"],
# )

# @router.get("/")
# async def get_accounts(
#     req: AccountSearchRequest = Depends(),
#     service: AuthService = Depends()
# ):
#     result = await service.search_accounts(req)
#     return APIResponse(
#         message="Get accounts",
#         status_code=status.HTTP_200_OK,
#         data=result
#     )

# @router.post("/sign-up")
# async def register_account(
#     dto: AccountCreateDTO,
#     service: AuthService = Depends()
# ):
#     result = await service.register_account(dto)
#     return APIResponse(
#         message="Get accounts",
#         status_code=status.HTTP_201_CREATED,
#         data=result
#     )

# @router.post("/sign-in")
# async def login_account(
#     dto: AccountLoginDTO,
#     service: AuthService = Depends()
# ):
#     result = await service.login_account(dto)
#     return APIResponse(
#         message="Login account",
#         status_code=status.HTTP_200_OK,
#         data= {
#             "access_token": result
#         }
#     )

# @router.put("/")
# async def edit_account(
#     id: str,
#     dto: AccountUpdateDTO,
#     service: AuthService = Depends()
# ):
#     result = await service.edit_account(id, dto)
#     return APIResponse(
#         message="Edit account",
#         status_code=status.HTTP_200_OK,
#         data=result
#     )

# @router.delete("/{id}")
# async def soft_delete_account(
#     id: str,
#     service: AuthService = Depends()
# ):
#     result = await service.delete_account(id)
#     return APIResponse(
#         message="Delete account",
#         status_code=status.HTTP_400_BAD_REQUEST,
#         data=result
#     )

from fastapi import APIRouter, Depends, FastAPI, status
from src.Features.AuthAPI.AccountDTO import AccountCreateDTO, AccountSearchRequest, AccountUpdateDTO, AccountLoginDTO
from src.Features.AuthAPI.AuthService import AuthService
from src.SharedKernel.base.APIResponse import APIResponse
from src.SharedKernel.persistence.Decorators import Controller

@Controller
class AuthController:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.router = APIRouter(
            prefix="/api/v1/auth",
            tags=["Auth"]
        )
        self.register_route()
        self.app.include_router(self.router)

    def register_route(self):
        
        @self.router.get("/")
        async def get_accounts(
            req: AccountSearchRequest = Depends(),
            service: AuthService = Depends()
        ):
            result = await service.search_accounts(req)
            return APIResponse(
                message="Get accounts",
                status_code=status.HTTP_200_OK,
                data=result
            )

        @self.router.post("/sign-up")
        async def register_account(
            dto: AccountCreateDTO,
            service: AuthService = Depends()
        ):
            result = await service.register_account(dto)
            return APIResponse(
                message="Account created successfully",
                status_code=status.HTTP_201_CREATED,
                data=result
            )

        @self.router.post("/sign-in")
        async def login_account(
            dto: AccountLoginDTO,
            service: AuthService = Depends()
        ):
            result = await service.login_account(dto)
            return APIResponse(
                message="Login successful",
                status_code=status.HTTP_200_OK,
                data={
                    "access_token": result
                }
            )

        @self.router.put("/{id}")
        async def edit_account(
            id: str,
            dto: AccountUpdateDTO,
            service: AuthService = Depends()
        ):
            result = await service.edit_account(id, dto)
            return APIResponse(
                message="Account updated successfully",
                status_code=status.HTTP_200_OK,
                data=result
            )

        @self.router.delete("/{id}")
        async def soft_delete_account(
            id: str,
            service: AuthService = Depends()
        ):
            result = await service.delete_account(id)
            return APIResponse(
                message="Account deleted successfully",
                status_code=status.HTTP_200_OK,
                data=result
        )
