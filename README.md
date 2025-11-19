[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/nwy6MBDZ)
# FAIR-LLM Installation Guide

## 🚀 Quick Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository (for demos)

```bash
git clone git@github.com:USAFA-AI-Center/fair_llm_demos.git
cd fair-llm-demos
```

### Step 2: Install All Dependencies

Simply install everything needed using the requirements file:

```bash
pip install -r requirements.txt
```

This will install:
- `fair-llm>=0.1` - The core FAIR-LLM package
- `python-dotenv` - For environment variable management
- `rich` - For beautiful terminal output
- `anthropic` - For Anthropic Claude integration
- `faiss-cpu` - For vector search capabilities
- `seaborn` - For data visualization
- `pytest` - For testing

### Step 3: Set Up API Keys

Create a `.env` file in your project root:

```bash
# Copy the example file
cp .env.example .env

# Or create a new one
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
echo "ANTHROPIC_API_KEY=your_anthropic_api_key_here" >> .env
```

Or export them as environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

### Step 4: Verify Installation

Run the verification script:

```bash
python verify_setup.py
```

You should see a colorful output showing all components are properly installed!

## 🎯 Running the Demos

Once installed, try the demo scripts:

### Essay Autograder Demo
```bash
# Basic grading
python demos/demo_committee_of_agents_essay_autograder.py \
  --essays essay_autograder_files/essays_to_grade/ \
  --rubric essay_autograder_files/grading_rubric.txt \
  --output essay_autograder_files/graded_essays/

# With RAG fact-checking
python demos/demo_committee_of_agents_essay_autograder.py \
  --essays essay_autograder_files/essays_to_grade/ \
  --rubric essay_autograder_files/grading_rubric.txt \
  --output essay_autograder_files/graded_essays/ \
  --materials essay_autograder_files/course_materials/
```

### Code Autograder Demo
```bash
# Static analysis only (safer)
python demos/demo_committee_of_agents_coding_autograder.py \
  --submissions coding_autograder_files/submissions/ \
  --rubric coding_autograder_files/rubric.txt \
  --output coding_autograder_files/reports/ \
  --no-run

# With test execution (requires sandbox)
python demos/demo_committee_of_agents_coding_autograder.py \
  --submissions coding_autograder_files/submissions/ \
  --tests coding_autograder_files/tests/test_calculator.py \
  --rubric coding_autograder_files/rubric.txt \
  --output coding_autograder_files/reports/
```

## 📦 Upgrading

To upgrade to the latest versions:

```bash
# Upgrade all packages
pip install --upgrade -r requirements.txt

# Or just upgrade fair-llm
pip install --upgrade fair-llm
```

## 🐛 Troubleshooting

### Missing Dependencies
If you get import errors, ensure all requirements are installed:
```bash
pip install -r requirements.txt --force-reinstall
```

### API Key Issues
The demos will create sample files if they don't exist, but ensure your API keys are set:
```python
python -c "import os; print('OpenAI Key:', 'Set' if os.getenv('OPENAI_API_KEY') else 'Not Set')"
```

### Virtual Environment Issues
Always use a virtual environment to avoid conflicts:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📚 What's Included

After installation, you'll have:
- ✅ The complete FAIR-LLM framework
- ✅ Multi-agent orchestration capabilities
- ✅ Document processing tools
- ✅ Vector search with FAISS
- ✅ Beautiful terminal output with Rich
- ✅ Complete demo applications

## 🎉 Next Steps

1. Run `python verify_setup.py` to confirm everything is working
2. Explore the `demos/` folder for examples
3. Set up and run some demos
4. Start building your own multi-agent demo files!

## 👥 Contributors
Developed by the USAFA AI Center team:

Ryan R (rrabinow@uccs.edu)
Austin W (austin.w@ardentinc.com)
Eli G (elijah.g@ardentinc.com)
Chad M (Chad.Mello@afacademy.af.edu)


# Design Choices (FOR FINAL SUBMISSION)

NOTE: football_agentv6.py is the final form of this pipeline and the one we plan to present in class.

Our project uses an agentic framework for evaluating Fantasy Football decisions. To build the model, we used the FairLLM framework with GPT-4 as the LLM. The framework is able to successfully gather information on players using the internet, does computation to evaluate relevant stats, and return a clear output that gives a verdict on the trade.

## Overall Goal

Goal of the project was to create a agentic framework able to evaluate Fantasy Football decisions using up to date information.

## Design Choices

This is a multi-agent framework using a manager that employs other agents to do the work necessary to solve the users query.

### Tool Description

The framework uses two tools built into the FairLLM framework:

1. WebSearcherTool: Accesses the internet to find relevant and up to date information on players given in the user query, including injury status, and performance metrics (yards/game, touchdowns, etc.)

2. SafeCalculatorTool: Able to do accurate math calculations using equations given by an agent. Used to find metrics to compare players, including Points-per-game deltas.

These two tools suffice to compare player performance to evaluate decisions.

Of note, we attempted to implement a vegas_tool (in vegas_tool.py) to implement betting odds into the decision making process but this did not work due to network restrictions. However, the WebSearcherTool is still able to deliver enough metrics to make a comparison.

### Agent Description
We created several agents as a part of the multi-agent framework:

1. researcher: Uses the WebSearcherTool to find current information on NFL players
2. analyst: Performs mathematical calculations to compare players
3. auditor: Examines and verifies research for consistency or missing sources, and delivers an overall confidence level.
4. synthesizer: Takes information from the researcher, analyst, and auditor, then produces the final output.
5. manager: Plans how to use the worker agents given the user query. 

### Pipeline

These are the steps in the models pipeline:

1. After user gives query, build tools and agents
2. Manager agent interprets user query and builds evaluation plan
3. worker agents are called in accordance with evaluation plan to gather information and synthesize a response. For example:
  - researcher called to gather information
  - auditor called to evaluate information gathered
  - analyst calculates metrics between players
  - synthesizer takes all information to create final response
4. Response returned to user

## Outputs and functionality

The pipeline is designed to handle several Fantasy Football Decisions:
1. trade
2. start_sit
3. hold_drop

The model outputs according to the following format:
Line 1: Verdict (HOLD|DROP|START|SIT|FAIR|FAVORABLE|UNFAVORABLE)
Line 2-4: Why - 2 to 4 sentences
Line 5: Quick stats
Line 6: Sources

## Limitations
- The model does not have access to fantasy rosters which might play a factor in some trade decisions
- The model was not designed to factor in more than two players at once. Trade decisions involving more than two players (e.g. 2 for 1) are possible but may not be accurate

## Known bugs
- Sometimes the output will display in a JSON format. While still readable, this is does not look good and is unintended. Steps taken to mitigate include instructing at all steps to not output JSON formats.

