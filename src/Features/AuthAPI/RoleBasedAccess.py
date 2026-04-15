from functools import wraps
from typing import List, Callable, Optional
from fastapi import Request
from src.Features.AuthAPI.JWTProvider import JWTProvider
from src.Domain.base_entities import AccountsRole
from src.SharedKernel.exception.APIException import APIException
from starlette import status


class RoleBasedAccess:
    """
    Class cung cấp decorator và helper functions cho role-based access control
    sử dụng JWT token.
    """
    
    def __init__(self, jwt_provider: Optional[JWTProvider] = None):
        """
        Initialize RoleBasedAccess với JWTProvider.
        
        Args:
            jwt_provider: Instance của JWTProvider. Nếu None, sẽ tạo mặc định.
        """
        self.jwt_provider = jwt_provider or JWTProvider()
    
    def require_role(self, *allowed_roles: AccountsRole):
        """
        Decorator kiểm tra role trong JWT token.
        
        Args:
            *allowed_roles: Danh sách roles được phép truy cập (AccountsRole enum values)
        
        Returns:
            Callable: Decorator function
        
        Raises:
            APIException: 401 nếu token không hợp lệ/hết hạn, 403 nếu role không được phép
        
        Usage:
            @self.role_access.require_role(AccountsRole.ADMIN)
            async def admin_only_endpoint(request: Request):
                pass
            
            @self.role_access.require_role(AccountsRole.ADMIN, AccountsRole.AGENT)
            async def staff_endpoint(request: Request):
                pass
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract request from args or kwargs
                request = self._get_request_from_args(args, kwargs)
                
                if not request:
                    raise APIException(
                        "Request object not found",
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                
                # Extract token from Authorization header
                token = self._extract_token_from_header(request)

                # Verify token and get payload
                payload = self._verify_token(token)
                
                # Check if role is allowed
                self._check_role(payload, list(allowed_roles))
                
                # Store user info in request state for later use
                request.state.user = {
                    "username": payload.get("username"),
                    "role": payload.get("role"),
                    "user_id": payload.get("user_id")
                }
                
                # Call original function
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def _get_request_from_args(self, args: tuple, kwargs: dict) -> Optional[Request]:
        """
        Helper method để extract Request object từ args hoặc kwargs.
        
        Args:
            args: Positional arguments từ decorated function
            kwargs: Keyword arguments từ decorated function
        
        Returns:
            Optional[Request]: Request object nếu tìm thấy, None nếu không
        """
        # Try to get from kwargs
        request = kwargs.get("request")
        if request and isinstance(request, Request):
            return request
        
        # Try to get from args
        for arg in args:
            if isinstance(arg, Request):
                return arg
        
        return None
    
    def _extract_token_from_header(self, request: Request) -> str:
        """
        Extract JWT token từ Authorization header.
        
        Expected format: "Bearer <token>"
        
        Args:
            request: FastAPI Request object
        
        Returns:
            str: JWT token string
        
        Raises:
            APIException: 401 nếu header không tồn tại hoặc format sai
        """
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            raise APIException(
                "Missing Authorization header",
                status_code=status.HTTP_401_UNAUTHORIZED
            )
        
        if not auth_header.startswith("Bearer "):
            raise APIException(
                "Invalid Authorization header format. Expected: Bearer <token>",
                status_code=status.HTTP_401_UNAUTHORIZED
            )
        
        # Extract token part (remove "Bearer ")
        token = auth_header[7:]
        
        if not token:
            raise APIException(
                "Empty token in Authorization header",
                status_code=status.HTTP_401_UNAUTHORIZED
            )
        
        return token
    
    def _verify_token(self, token: str) -> dict:
        """
        Verify JWT token và return payload.
        
        Args:
            token: JWT token string
        
        Returns:
            dict: Token payload chứa username, role, user_id, exp
        
        Raises:
            APIException: 401 nếu token không hợp lệ hoặc hết hạn
        """
        payload = self.jwt_provider.verify_token(token)
        
        if payload is None:
            raise APIException(
                "Invalid or expired token",
                status_code=status.HTTP_401_UNAUTHORIZED
            )
        
        return payload
    
    def _check_role(self, payload: dict, allowed_roles: List[AccountsRole]):
        """
        Kiểm tra role trong payload với danh sách roles cho phép.
        
        Args:
            payload: JWT token payload
            allowed_roles: Danh sách AccountsRole được phép
        
        Raises:
            APIException: 401 nếu role không tồn tại trong token, 403 nếu role không được phép
        """
        user_role = payload.get("role")
        
        if not user_role:
            raise APIException(
                "Role not found in token",
                status_code=status.HTTP_401_UNAUTHORIZED
            )
        
        # Convert string to enum nếu cần
        if isinstance(user_role, str):
            try:
                user_role = AccountsRole(user_role)
            except ValueError:
                raise APIException(
                    f"Invalid role in token: {user_role}",
                    status_code=status.HTTP_401_UNAUTHORIZED
                )
        
        # Check if role is in allowed list
        if user_role not in allowed_roles:
            raise APIException(
                f"Access denied. Required roles: {[r.value for r in allowed_roles]}. "
                f"Current role: {user_role.value if isinstance(user_role, AccountsRole) else user_role}",
                status_code=status.HTTP_403_FORBIDDEN
            )


def get_current_user(request: Request) -> dict:
    """
    Helper function để lấy user info từ request state.
    Phải được gọi sau khi decorator @require_role() đã được thực thi.
    
    Args:
        request: FastAPI Request object
    
    Returns:
        dict: User info với keys: username, role, user_id
    
    Raises:
        APIException: 401 nếu user chưa được authenticated
    
    Example:
        @self.role_access.require_role(AccountsRole.ADMIN)
        async def protected_endpoint(request: Request):
            user = get_current_user(request)
            return {"message": f"Welcome {user['username']}"}
    """
    if not hasattr(request.state, "user"):
        raise APIException(
            "User not authenticated. Please use @require_role decorator first.",
            status_code=status.HTTP_401_UNAUTHORIZED
        )
    
    return request.state.user


def get_current_role(request: Request) -> AccountsRole:
    """
    Helper function để lấy role hiện tại của user từ request state.
    Phải được gọi sau khi decorator @require_role() đã được thực thi.
    
    Args:
        request: FastAPI Request object
    
    Returns:
        AccountsRole: Role của user hiện tại
    
    Raises:
        APIException: 401 nếu user chưa được authenticated
    
    Example:
        @self.role_access.require_role(AccountsRole.ADMIN, AccountsRole.AGENT)
        async def staff_endpoint(request: Request):
            role = get_current_role(request)
            if role == AccountsRole.ADMIN:
                # Admin-specific logic
                pass
            return {"role": role.value}
    """
    user = get_current_user(request)
    role = user.get("role")
    
    if isinstance(role, str):
        return AccountsRole(role)
    
    return role


def get_current_user_id(request: Request) -> str:
    """
    Helper function để lấy user_id hiện tại từ request state.
    Phải được gọi sau khi decorator @require_role() đã được thực thi.
    
    Args:
        request: FastAPI Request object
    
    Returns:
        str: User ID (UUID string)
    
    Raises:
        APIException: 401 nếu user chưa được authenticated
    
    Example:
        @self.role_access.require_role(AccountsRole.CUSTOMER)
        async def get_my_tickets(request: Request):
            user_id = get_current_user_id(request)
            return await ticket_service.get_tickets_by_user(user_id)
    """
    user = get_current_user(request)
    user_id = user.get("user_id")
    
    if not user_id:
        raise APIException(
            "User ID not found in token",
            status_code=status.HTTP_401_UNAUTHORIZED
        )
    
    return user_id
