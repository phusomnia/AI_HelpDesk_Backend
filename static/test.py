from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def init_schema(driver):
    """Create constraints and indexes"""
    with driver.session() as session:
        # Create unique constraints
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE")
        
        # Create indexes for common queries
        session.run("CREATE INDEX IF NOT EXISTS FOR (u:User) ON (u.email)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (p:Post) ON (p.createdAt)")
        
        print("✓ Schema initialized")
