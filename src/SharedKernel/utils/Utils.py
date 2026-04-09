from SharedKernel.base.Logger import get_logger

logger = get_logger(__name__)

class Utils:
    @staticmethod
    def generate_conversation_key(id_a: str, id_b: str):
        sorted_ids = sorted([str(id_a), str(id_b)])
        
        combined_string = f"dm_{sorted_ids[0]}_{sorted_ids[1]}"

        return combined_string

    
    @staticmethod
    def extract_customer_id_from_conversation_key(conversation_key: str, agent_id: str):
        key_part = conversation_key.replace("dm", "")
        ids = key_part.split("_")
        logger.info(ids)
        
        for id in ids:
            if id != agent_id and id != "":
                return id
        
        return None