class QueryExtension:
    def __init__(self, base_sql: str):
        self._base = base_sql
        self._filters = []
        self._params = {}
        self._limit = ""  
        self._order = ""

    def filter(self, condition: bool, clause: str, **params):
        if condition:
            self._filters.append(clause)
            self._params.update(params)
        return self

    def range_filter(self, column: str, min_val, max_val):
        if min_val is not None and max_val is not None:
            self._filters.append(f"{column} BETWEEN :min AND :max")
            self._params["min"] = min_val
            self._params["max"] = max_val
        return self

    def order_by(self, clause: str):
        self._order = f" ORDER BY {clause}"
        return self

    def paginate(self, page: int, size: int):
        self._params["limit"] = size
        self._params["offset"] = (page - 1) * size
        self._limit = " LIMIT :limit OFFSET :offset"
        return self

    def build_select(self, select_clause: str):
        where_clause = ""
        if self._filters:
            where_clause = " AND " + " AND ".join(self._filters)

        sql = f"""
        SELECT {select_clause}
        {self._base}
        {where_clause}
        {self._limit}
        """

        return sql, self._params

    
    def build_count(self):
        where_clause = ""
        if self._filters:
            where_clause = " AND " + " AND ".join(self._filters)

        count_params = {k: v for k, v in self._params.items() 
                   if k not in ['limit', 'offset']}

        sql = f"""
        SELECT COUNT(*) AS total
        {self._base}
        {where_clause}
        """
        return sql, count_params