def main():    
    import pandas as pd
    import hashlib
    import json
    from time import time
    import random

    # Replace 'your_file.xlsx' with the path to your Excel file
    excel_file = 'delivery_truck_trips.xlsx'
    df = pd.read_excel(excel_file)

    class Block:
        def __init__(self, index, transactions, timestamp, previous_hash, creator, creator_stake):
            self.index = index
            self.transactions = transactions
            self.timestamp = timestamp
            self.previous_hash = previous_hash
            self.creator = creator  # Add creator of the block
            self.creator_stake = creator_stake  # Add creator's stake
            self.hash = self.compute_hash()

        def compute_hash(self):
            block_string = json.dumps(self.__dict__, sort_keys=True)
            return hashlib.sha256(block_string.encode()).hexdigest()

    class Blockchain:
        def __init__(self):
            self.chain = []
            self.create_genesis_block()

        def create_genesis_block(self):
            genesis_block = Block(0, [], str(time()), "0", "Genesis", 0)  # Creator is "Genesis" with 0 stake
            self.chain.append(genesis_block)

        def add_block(self, transactions):
            last_block = self.chain[-1]
            creator = random.choice(list(stakeholder_stakes.keys()))  # Choose a random stakeholder
            creator_stake = stakeholder_stakes[creator]  # Get the stake of the chosen stakeholder
            new_block = Block(index=last_block.index + 1,
                            transactions=transactions,
                            timestamp=str(time()),
                            previous_hash=last_block.hash,
                            creator=creator,
                            creator_stake=creator_stake)
            self.chain.append(new_block)

        def is_valid(self):
            for i in range(1, len(self.chain)):
                current_block = self.chain[i]
                previous_block = self.chain[i - 1]

                # Recompute the hash of the current block
                if current_block.hash != current_block.compute_hash():
                    return False

                # Check if the previous hash matches
                if current_block.previous_hash != previous_block.hash:
                    return False
                
                # Check if creator's stake is sufficient
                if current_block.creator_stake < 100:  # Example: Require a minimum stake of 100 for block creation
                    return False
                    
            return True

    # Initialize blockchain
    my_blockchain = Blockchain()

    # Initialize stakeholders with random stakes (for demonstration)
    stakeholders = ["Stakeholder1", "Stakeholder2", "Stakeholder3"]
    stakeholder_stakes = {stakeholder: random.randint(50, 500) for stakeholder in stakeholders}

    # Iterate over DataFrame rows as (index, Series) pairs
    for index, row in df.iterrows():
        trip_data = {
            'BookingID': row['BookingID'],
            'Origin_Location': row['Origin_Location'],
            'Destination_Location': row['Destination_Location']
        }
        my_blockchain.add_block(trip_data)

    # Verify blockchain integrity
    print("Blockchain valid?", my_blockchain.is_valid())

    # Print all blocks in the blockchain
    for block in my_blockchain.chain:
        print("Block Index:", block.index)
        print("Transactions:", block.transactions)
        print("Timestamp:", block.timestamp)
        print("Previous Hash:", block.previous_hash)
        print("Creator:", block.creator)
        print("Creator Stake:", block.creator_stake)
        print("Hash:", block.hash)
        print()

main()
