"""
LangChain agent setup with LoRA model support.
Creates LangChain agents that can use FinQA tools with fine-tuned models.
"""

from typing import Optional, Tuple
try:
    from langchain_community.llms import HuggingFacePipeline
except ImportError:
    try:
        from langchain.llms import HuggingFacePipeline
    except ImportError:
        # Fallback for newer versions
        HuggingFacePipeline = None

try:
    from langchain.agents import AgentType, initialize_agent, AgentExecutor
except ImportError:
    # Try alternative imports for newer versions
    try:
        from langchain.agents import AgentExecutor
        AgentType = None
        initialize_agent = None
    except ImportError:
        AgentExecutor = None
        AgentType = None
        initialize_agent = None

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enhanced_executor import EnhancedExecutor
from tool_calling.langchain_tools import create_finqa_langchain_tools


def create_langchain_llm_with_lora(
    model_name: str,
    adapter_path: Optional[str] = None,
    device: str = "auto",
    load_in_8bit: bool = False,
    torch_dtype: str = "bfloat16",
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    do_sample: bool = True
) -> Tuple[HuggingFacePipeline, AutoModelForCausalLM, AutoTokenizer]:
    """
    Create LangChain LLM from HuggingFace model with optional LoRA adapters.
    
    Args:
        model_name: Base model name (e.g., "meta-llama/Meta-Llama-3-8B-Instruct")
        adapter_path: Path to LoRA adapters (e.g., "results/lora/Meta-Llama-3-8B-Instruct/final_model")
        device: Device to use ("auto", "cuda", "cpu")
        load_in_8bit: Whether to load in 8-bit
        torch_dtype: Torch dtype ("bfloat16" or "float16")
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        
    Returns:
        Tuple of (LangChain LLM, model, tokenizer)
    """
    print(f"Loading base model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load base model
    dtype = getattr(torch, torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        load_in_8bit=load_in_8bit,
        trust_remote_code=True
    )
    
    # Load LoRA adapters if provided
    if adapter_path:
        print(f"Loading LoRA adapters from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        # Keep adapters separate (can unload later if needed)
        model = model.eval()
    else:
        model.eval()
    
    print(f"Model loaded on device: {next(model.parameters()).device}")
    
    # Create HuggingFace pipeline
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        return_full_text=False,
        device_map=device if device != "auto" else None
    )
    
    # Wrap in LangChain
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return llm, model, tokenizer


def create_finqa_agent_with_lora(
    executor: EnhancedExecutor,
    model_name: str,
    adapter_path: Optional[str] = None,
    load_in_8bit: bool = False,
    max_iterations: int = 10,
    verbose: bool = True
):
    """
    Create LangChain agent with FinQA tools and LoRA model.
    
    Args:
        executor: EnhancedExecutor instance with table context
        model_name: Base model name
        adapter_path: Path to LoRA adapters (optional)
        load_in_8bit: Whether to load in 8-bit
        max_iterations: Maximum tool calling iterations
        verbose: Whether to print agent actions
        
    Returns:
        Tuple of (agent, model, tokenizer)
    """
    # Get tools
    tools = create_finqa_langchain_tools(executor)
    print(f"Created {len(tools)} FinQA tools")
    
    # Create LLM with LoRA
    llm, model, tokenizer = create_langchain_llm_with_lora(
        model_name,
        adapter_path,
        load_in_8bit=load_in_8bit
    )
    
    # Create agent
    print("Creating LangChain agent...")
    
    if initialize_agent is not None and AgentType is not None:
        # Use classic API - try STRUCTURED_CHAT first (better for structured tools)
        try:
            agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=verbose,
                return_intermediate_steps=True,
                max_iterations=max_iterations,
                early_stopping_method="generate"
            )
            print("✓ Using STRUCTURED_CHAT agent")
        except Exception as e:
            print(f"Error with STRUCTURED_CHAT: {e}")
            # Fallback to ZERO_SHOT_REACT_DESCRIPTION
            try:
                agent = initialize_agent(
                    tools=tools,
                    llm=llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=verbose,
                    return_intermediate_steps=True,
                    max_iterations=max_iterations,
                    early_stopping_method="generate"
                )
                print("✓ Using ZERO_SHOT_REACT_DESCRIPTION agent")
            except Exception as e2:
                print(f"Error with ZERO_SHOT_REACT_DESCRIPTION: {e2}")
                raise
    else:
        raise ImportError("LangChain classic API not available. Please install langchain<0.1")
    
    # Ensure agent is AgentExecutor and has return_intermediate_steps enabled
    # Note: initialize_agent already returns AgentExecutor, so this check is mainly for safety
    try:
        from langchain.agents import AgentExecutor
        if not isinstance(agent, AgentExecutor):
            # This shouldn't happen with initialize_agent, but handle it if it does
            print("Warning: Agent is not AgentExecutor, but should be from initialize_agent")
    except ImportError:
        pass  # AgentExecutor not available
    
    print("Agent created successfully!")
    return agent, model, tokenizer

