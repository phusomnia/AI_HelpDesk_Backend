from fastapi import APIRouter, Depends, FastAPI, Request, status
from src.Domain.base_entities import AccountsRole
from src.Features.AuthAPI.AccountDTO import CreateAccountRequest, LoginAccountRequest, SearchAccountRequest, UpdateAccountRequest
from src.Features.AuthAPI.AuthService import AuthService
from src.Features.AuthAPI.RoleBasedAccess import RoleBasedAccess, get_current_user, get_current_role, get_current_user_id
from src.SharedKernel.base.APIResponse import APIResponse
from src.SharedKernel.exception.APIException import APIException
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

        @self.router.get("/access", description="Get redirect URL based on user role")
        async def access(
            request: Request
        ):
            try:
                auth_header = request.headers.get("Authorization")
                if not auth_header or not auth_header.startswith("Bearer "):
                    raise APIException(
                        "Missing or invalid Authorization header",
                        status_code=status.HTTP_401_UNAUTHORIZED
                    )

                token = auth_header[7:]
                payload = self.role_access._verify_token(token)

                user_role_str = payload.get("role")
                if not user_role_str:
                    raise APIException(
                        "Role not found in token",
                        status_code=status.HTTP_401_UNAUTHORIZED
                    )

                try:
                    user_role = AccountsRole(user_role_str)
                except ValueError:
                    raise APIException(
                        f"Invalid role in token: {user_role_str}",
                        status_code=status.HTTP_401_UNAUTHORIZED
                    )

                # Determine redirect URL based on role
                if user_role == AccountsRole.CUSTOMER:
                    redirect_url = "/user_portal"
                elif user_role in [AccountsRole.AGENT, AccountsRole.ADMIN]:
                    redirect_url = "/management"
                else:
                    raise APIException(
                        f"Unknown role: {user_role_str}",
                        status_code=status.HTTP_403_FORBIDDEN
                    )

                return APIResponse(
                    message="Access granted",
                    status_code=status.HTTP_200_OK,
                    data={
                        "redirect_url": redirect_url,
                        "role": user_role.value,
                        "username": payload.get("username"),
                        "user_id": payload.get("user_id")
                    }
                )

            except APIException as e:
                raise e
            except Exception as e:
                raise APIException(
                    f"Error processing access request: {str(e)}",
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                )


