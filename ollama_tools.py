import json
import traceback
from typing import Dict, Union
from cat.mad_hatter.decorators import hook
from cat.log import log
from langchain.schema import AgentAction
from .utils import format_agent_input
from cat.mad_hatter.decorators.tool import CatTool
from cat.looking_glass.callbacks import NewTokenHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from cat.looking_glass import prompts


def choose_tool(cat, agent_input, allowed_tools) -> tuple[CatTool, str]:

    allowed_tools_names = [t.name for t in allowed_tools]

    tools_str = "\n".join(
        [f"> {tool.name}: {tool.description}" for tool in allowed_tools])

#     prompt = f"""Chose one of the following tools to respond to the user reqest:

# {tools_str}
# none_of_the_others: none_of_the_others(None) - Use this tool if none of the others tools help. Input is always None.

# Human: What time is it?
# Tool: get_the_time

# {agent_input["chat_history"]}
# Human: {agent_input["input"]}
# Tool: """

    prompt = f"""You are Assistant. Assistant is a expert JSON builder designed to assist with a wide range of tools.
    
Assistant is able to select tools using JSON strings that contain "action" and "action_input" parameters.

All of Assistant's communication is performed using this JSON format. The assistant NEVER outputs anything other than a json object with an action and action_input fields!

Tools available to Assistant are:

{tools_str}
> none_of_the_others: none_of_the_others(None) - Use this tool if none of the others tools help. Input is always None.

Here is an example of a previous conversation between User and Assistant:
---

User: Hey how are you today?
Assistant: {{"action": "none_of_the_others","action_input": "I'm good thanks, how are you?"}}

User: 7
Assistant: {{"action": "none_of_the_others","action_input": "It looks like the answer is 7!"}}

User: September 21, 2022: Olivia Wilde says kids\' happiness remains top priority for her and Jason Sudeikis. During an appearance on The Kelly Clarkson Show, Wilde talked about the ups and downs of ... Olivia Wilde is "quietly dating again" following her November 2022 split from. Harry Styles, a source exclusively tells Life & Style. "The man she\'s with is \'normal\' by Hollywood ... Wilde honored Sudeikis with a sweet tribute on his 42nd birthday. "I have approximately one billion pictures of this guy, my partner in life-crime, who was born on this day in 1975, but this one ... November 18, 2022: Olivia Wilde and Harry Styles are "taking a break". After nearly two years together, Wilde and Styles are pressing pause on their romance. Multiple sources confirmed exclusively ... Wilde told Allure in October 2013 that when she first met Sudeikis she "thought he was so charming." "He\'s a great dancer, and I\'m a sucker for great dancers," she said at the time. "But he didn\'t ...
Assistant: {{"action": "none_of_the_others", "action_input": "Jason Sudeikis"}}

User: What time is it?
Assistant: {{"action": "get_the_time", "action_input": "None"}}

User: What is the weather like in New York?
Assistant: {{"action": "none_of_the_others", "action_input": "I'm not sure, let me check"}}

---
Assistant is very intelligent and knows it's limitations, so it will always try to use a tool when applicable, even if Assistant thinks it knows the answer!
Assistant will only use the available tools and NEVER a tool not listed. If the User's question does not require the use of a tool, Assistant will use the "none_of_the_others" action to give a normal response.
Respond to the following in JSON with 'action' and 'action_input' values. Once you have outputted the JSON, stop outputting and output nothing else!
User: {agent_input["input"]}
Assistant:"""

    output = cat.llm(prompt)
    output = output.strip().replace("\\", "")

    log.critical("TOOL CHOOSED")

    try:
        output_json = json.loads(output)
    except Exception as e:
        log.error("Error in tool choosing")
        log.error(e)
        return None

    log.critical(output_json)

    for t in allowed_tools:
        if t.name in output_json["action"]:
            log.critical("wa")
            return (t, output_json["action_input"])

    return None


def extract_input(cat, agent_input, choosed_tool):
    prompt = f"""Given the following tool, extract the input from the following messages:

get_the_time: get_the_time(tool_input) - Replies to "what time is it", "get the clock" and similar questions. Input is always None.
        
Human: What time is it?
Input: None

{choosed_tool.name}: {choosed_tool.description}   

{agent_input["chat_history"]}
Human: {agent_input["input"]}
Input: """

    output = cat.llm(prompt)
    output = output.strip()

    log.debug("TOOL INPUT")
    print(output)
    return output


def execute_tool_agent(cat, agent_input, allowed_tools):
    default_return = {"intermediate_steps": [], "output": None}

    try:
        chosen_tool = choose_tool(cat, agent_input, allowed_tools)

        if chosen_tool is None:
            return default_return
        else:
            tool, tool_input = chosen_tool

        # tool_input = extract_input(cat, agent_input, tool)

        output_tool = tool.run(tool_input)

        return {
            "intermediate_steps": [
                (
                    AgentAction(tool=tool.name,
                                tool_input=tool_input, log=f"Tool chosen: {tool.name} with Input: {tool_input}"),
                    output_tool,
                )
            ],
            "output": output_tool,
        }
    except Exception as e:
        log.error("Error in tool execution")
        traceback.print_stack()
        print(e)
        return default_return


@hook
def agent_fast_reply(fast_reply: Dict, cat) -> Union[Dict, None]:
    agent_input = format_agent_input(cat.working_memory)
    # tools currently recalled in working memory
    recalled_tools = cat.working_memory["procedural_memories"]
    # Get the tools names only
    tools_names = [t[0].metadata["name"] for t in recalled_tools]
    allowed_tools = [i for i in cat.mad_hatter.tools if i.name in tools_names]

    prompt_prefix = cat.mad_hatter.execute_hook(
        "agent_prompt_prefix", prompts.MAIN_PROMPT_PREFIX, cat=cat)
    prompt_suffix = cat.mad_hatter.execute_hook(
        "agent_prompt_suffix", prompts.MAIN_PROMPT_SUFFIX, cat=cat)

    tool_result = execute_tool_agent(cat, agent_input, allowed_tools)
    print(tool_result)

    try:
        used_tools = None
        # If tools_result["output"] is None the LLM has used the fake tool none_of_the_others
        # so no relevant information has been obtained from the tools.
        if tool_result["output"] is not None:

            # Extract of intermediate steps in the format ((tool_name, tool_input), output)
            used_tools = list(map(lambda x: (
                (x[0].tool, x[0].tool_input), x[1]), tool_result["intermediate_steps"]))

            # Get the name of the tools that have return_direct
            return_direct_tools = [t for t in allowed_tools if t.return_direct]

            # execute_tool_agent returns immediately when a tool with return_direct is called,
            # so if one is used it is definitely the last one used
            if used_tools[-1][0][0] in return_direct_tools:
                # intermediate_steps still contains the information of all the tools used even if their output is not returned
                tool_result["intermediate_steps"] = used_tools
                return tool_result

            # Adding the tools_output key in agent input, needed by the memory chain
            agent_input["tools_output"] = "## Tools output: \n" + \
                tool_result["output"] if tool_result["output"] else ""
        else:
            # If no relevant information has been obtained from the tools, the tools_output key is not added to the agent input
            agent_input["tools_output"] = ""

        # Execute the memory chain
        out = execute_memory_chain(
            agent_input, prompt_prefix, prompt_suffix, cat)

        # If some tools are used the intermediate step are added to the agent output
        out["intermediate_steps"] = used_tools

        return out
    except Exception as e:
        log.error(e)


def execute_memory_chain(agent_input, prompt_prefix, prompt_suffix, stray):

    input_variables = [i for i in agent_input.keys(
    ) if i in prompt_prefix + prompt_suffix]
    # memory chain (second step)
    memory_prompt = PromptTemplate(
        template=prompt_prefix + prompt_suffix,
        input_variables=input_variables
    )

    memory_chain = LLMChain(
        prompt=memory_prompt,
        llm=stray._llm,
        verbose=True
    )

    out = memory_chain(agent_input, callbacks=[NewTokenHandler(stray)])
    out["output"] = out["text"]
    del out["text"]
    return out
