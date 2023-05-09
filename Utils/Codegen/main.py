from Utils.Codegen.codegen import set_seed, print_time, sample, truncate

# @markdown # Try out the model
rng_seed = 42  # @param {type:"integer"}
rng_deterministic = True  # @param {type:"boolean"}
p = 0.95  # @param {type:"number"}
t = 0.2  # @param {type:"number"}
max_length = 128  # @param {type:"integer"}
batch_size = 1  # @param {type:"integer"}
context = "def hello_world():"  # @param {type:"string"}

set_seed(rng_seed, deterministic=rng_deterministic)

# (4) sample

with print_time('sampling'):
    completion = sample(device=device, model=model, tokenizer=tokenizer, context=context, pad_token_id=pad,
                        num_return_sequences=batch_size, temp=t, top_p=p, max_length_sample=max_length)[0]
    truncation = truncate(completion)

    print('=' * 100)
    print(completion)
    print('=' * 100)
    print(context + truncation)
    print('=' * 100)

# !python -m jaxformer.hf.sample --model $chosen_model \
#                  --rng-seed $rng_seed \
#                  --p $p \
#                  --t $t \
#                  --max-length $max_length \
#                  --batch-size $batch_size \
#                  --context '$context'