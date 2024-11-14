import unittest
import HMM

class TestHMM(unittest.TestCase):
    def test_load(self):
        # Assuming load function takes a file path as input and returns some data structure
        test_file_path = 'test_data.txt'
        
        # Create a test file with some content
        with open(test_file_path, 'w') as f:
            f.write('some test data')

        h1 = HMM.HMM()

        # Call the load function
        h1.load("cat")

        expected_emissions = {
            'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
            'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
            'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}
        }

        expected_transitions = {
            '#': {'happy': '0.5', 'grumpy': '0.5', 'hungry': '0'},
            'happy': {'happy': '0.5', 'grumpy': '0.1', 'hungry': '0.4'},
            'grumpy': {'happy': '0.6', 'grumpy': '0.3', 'hungry': '0.1'},
            'hungry': {'happy': '0.1', 'grumpy': '0.6', 'hungry': '0.3'}
        }

        # Check if the result is as expected
        self.assertEqual(h1.emissions, expected_emissions)  
        self.assertEqual(h1.transitions, expected_transitions)  

if __name__ == '__main__':
    unittest.main()