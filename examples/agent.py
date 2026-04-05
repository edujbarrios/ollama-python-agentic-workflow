from ollama import Agent


def add_two_numbers(a: int, b: int) -> int:
  """
  Add two numbers together.

  Args:
    a (int): The first number
    b (int): The second number

  Returns:
    int: The sum of the two numbers
  """
  return int(a) + int(b)


def subtract_two_numbers(a: int, b: int) -> int:
  """
  Subtract two numbers.

  Args:
    a (int): The first number
    b (int): The second number

  Returns:
    int: The difference of the two numbers
  """
  return int(a) - int(b)


# Create an agent with tools - it handles the tool-calling loop automatically
agent = Agent(
  model='llama3.1',
  tools=[add_two_numbers, subtract_two_numbers],
  system='You are a helpful math assistant.',
)

# First question
response = agent.chat('What is three plus one?')
print('Response:', response.message.content)

# Follow-up question - the agent maintains conversation history
response = agent.chat('Now subtract 2 from that result.')
print('Follow-up:', response.message.content)

# View conversation history
print('\nConversation history:')
for msg in agent.messages:
  role = msg.role if hasattr(msg, 'role') else msg.get('role')
  content = msg.content if hasattr(msg, 'content') else msg.get('content')
  if role != 'system':
    print(f'  [{role}] {content}')
