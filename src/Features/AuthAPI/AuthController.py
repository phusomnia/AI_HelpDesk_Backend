from fastapi import APIRouter, Depends, FastAPI, Request, status
from src.Domain.base_entities import AccountsRole
from src.Features.AuthAPI.AccountDTO import CreateAccountRequest, LoginAccountRequest, SearchAccountRequest, UpdateAccountRequest
from src.Features.AuthAPI.AuthService import AuthService
from src.Features.AuthAPI.RoleBasedAccess import RoleBasedAccess, get_current_user, get_current_role, get_current_user_id
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
        self.role_access = RoleBasedAccess()
        self.register_route()
        self.app.include_router(self.router)

    def register_route(self):
        
        @self.router.get("/account")
        async def get_accounts(
            req: SearchAccountRequest = Depends(),
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
            dto: CreateAccountRequest,
            service: AuthService = Depends()
        ):
            result = await service.register_account(dto)
            return APIResponse(
                message="Account created successfully",
                status_code=status.HTTP_201_CREATED,
                data=result
            )

        @self.router.post("/sign-in", description="Login account")
        async def login_account(
            dto: LoginAccountRequest,
            service: AuthService = Depends()
        ):
            result = await service.login_account(dto)
            return APIResponse(
                message="Login successfully",
                status_code=status.HTTP_200_OK,
                data={
                    "access_token": result
                }
            )

        @self.router.get("/account/{id}", description="Get user by ID")
        async def get_user_by_id(
            id: str,
            service: AuthService = Depends()
        ):
            result = await service.get_user_by_id(id)
            return APIResponse(
                message="User retrieved successfully",
                status_code=status.HTTP_200_OK,
                data=result
            )

        @self.router.put("/account/{id}", description="Update account")
        async def edit_account(
            id: str,
            dto: UpdateAccountRequest,
            service: AuthService = Depends()
        ):
            result = await service.edit_account(id, dto)
            return APIResponse(
                message="Account updated successfully",
                status_code=status.HTTP_200_OK,
                data=result
            )

        @self.router.delete("/account/{id}", description="Soft delete account")
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

        # Role-based access control demo endpoints
        @self.router.get("/admin-only", description="Admin only endpoint demo")
        @self.role_access.require_role(AccountsRole.ADMIN)
        async def admin_only_endpoint(
            request: Request
        ):
            user = get_current_user(request)
            return APIResponse(
                message="Welcome Admin!",
                status_code=status.HTTP_200_OK,
                data={
                    "username": user["username"],
                    "role": user["role"],
                    "user_id": user["user_id"]
                }
            )
