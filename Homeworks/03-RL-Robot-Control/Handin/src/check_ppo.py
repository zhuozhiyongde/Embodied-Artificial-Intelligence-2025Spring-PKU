def check_monte_carlo(monte_carlo_advantage):
    import numpy as np

    point = 0
    np.random.seed(0)
    rewards = np.random.rand(5)
    values = np.random.rand(6)
    gamma = 0.99
    advantages1 = monte_carlo_advantage(rewards, values, gamma)
    if advantages1 is None:
        raise NotImplementedError

    sol1 = np.array([2.13738597e+00, 1.81944973e+00, 6.65648795e-01, 6.38673841e-04,
       4.02132805e-02])
    if np.allclose(advantages1, sol1):
        point += 1
    else:
        f"Expected {sol1}, but got {advantages1}"

    # test 2
    rewards = np.random.rand(5)
    values = np.random.rand(6)
    gamma = 0.99
    advantages2 = monte_carlo_advantage(rewards, values, gamma)
    sol2 = np.array([ 2.13084018,  0.8059293 ,  0.30316101, -0.71271808, -0.89148904])
    if np.allclose(advantages2, sol2):
        point += 1
    else:
        f"Expected {sol2}, but got {advantages2}"

    # test 3
    rewards = np.random.rand(5)
    values = np.random.rand(6)
    gamma = 0.99
    advantages3 = monte_carlo_advantage(rewards, values, gamma)
    sol3 = np.array([ 1.16407442,  1.14205468,  0.47763485,  0.51728516, -0.6308804 ])
    if np.allclose(advantages3, sol3):
        point += 1
    else:
        f"Expected {sol3}, but got {advantages3}"

    print(f"Monte Carlo total points: {point}/3")


def check_td_residual(td_residual_advantage):
    import numpy as np
    point = 0
    np.random.seed(0)
    rewards = np.random.rand(5)
    values = np.random.rand(6)
    gamma = 0.99
    advantages1 = td_residual_advantage(rewards, values, gamma)
    if advantages1 is None:
        raise NotImplementedError
    sol1 = np.array([ 0.33613073,  1.16045743,  0.66501651, -0.03917247,  0.82402107])
    if np.allclose(advantages1, sol1):
        point += 1
    else:
        print(f"Expected {sol1}, but got {advantages1}")

    # test 2
    rewards = np.random.rand(5)
    values = np.random.rand(6)
    gamma = 0.99
    advantages2 = td_residual_advantage(rewards, values, gamma)
    sol2 = np.array([ 1.33297017,  0.5057999 ,  1.00875191,  0.16985607, -0.10032206])
    if np.allclose(advantages2, sol2):
        point += 1
    else:
        print(f"Expected {sol2}, but got {advantages2}")

    # test 3
    rewards = np.random.rand(5)
    values = np.random.rand(6)
    gamma = 0.99
    advantages3 = td_residual_advantage(rewards, values, gamma)
    sol3 = np.array([ 0.03344028,  0.66919618, -0.03447746,  1.14185676, -0.17929157])
    if np.allclose(advantages3, sol3):
        point += 1
    else:
        print(f"Expected {sol3}, but got {advantages3}")

    print(f"TD residual total points: {point}/3")


def check_gae(generalized_advantage_estimation):
    import numpy as np
    point = 0
    np.random.seed(0)
    rewards = np.random.rand(5)
    values = np.random.rand(6)
    gamma = 0.99
    lam = 0.95
    advantages1 = generalized_advantage_estimation(rewards, values, gamma, lam)
    if advantages1 is None:
        raise NotImplementedError
    sol1 = np.array([2.62791035, 2.43676728, 1.3570546 , 0.73581934, 0.82402107])
    if np.allclose(advantages1, sol1):
        point += 1
    else:
        print(f"Expected {sol1}, but got {advantages1}")

    # test 2
    rewards = np.random.rand(5)
    values = np.random.rand(6)
    gamma = 0.99
    lam = 0.95
    advantages2 = generalized_advantage_estimation(rewards, values, gamma, lam)
    sol2 = np.array([ 2.76376849,  1.52131666,  1.07976264,  0.07550317, -0.10032206])
    if np.allclose(advantages2, sol2):
        point += 1
    else:
        print(f"Expected {sol2}, but got {advantages2}")

    # test 3
    rewards = np.random.rand(5)
    values = np.random.rand(6)
    gamma = 0.99
    lam = 0.95
    advantages3 = generalized_advantage_estimation(rewards, values, gamma, lam)
    sol3 = np.array([ 1.44196499,  1.49763392,  0.88084821,  0.97323304, -0.17929157])
    if np.allclose(advantages3, sol3):
        point += 1
    else:
        print(f"Expected {sol3}, but got {advantages3}")

    print(f"GAE total points: {point}/3")


def check_policy_loss(policy_loss_fn):
    import numpy as np
    point = 0
    np.random.seed(0)
    ratio = np.random.uniform(0.8, 1.2, 5)
    adv = np.random.rand(5)
    dist_entropy = 0.1
    epsilon = 0.2
    entropy_weight = 0.01
    policy_loss1 = policy_loss_fn(ratio, adv, dist_entropy, epsilon, entropy_weight)
    if policy_loss1 is None:
        raise NotImplementedError
    sol1 = -0.6839767133779934
    if np.allclose(policy_loss1, sol1):
        point += 1
    else:
        print(f"Expected {sol1}, but got {policy_loss1}")


    # test 2
    ratio = np.random.uniform(0.3, 1.5, 5)
    adv = np.random.rand(5)
    dist_entropy = 0.2
    epsilon = 0.3
    entropy_weight = 0.02
    policy_loss2 = policy_loss_fn(ratio, adv, dist_entropy, epsilon, entropy_weight)
    sol2 = -0.46238592009535634

    if np.allclose(policy_loss2, sol2):
        point += 1
    else:
        print(f"Expected {sol2}, but got {policy_loss2}")


    # test 3
    ratio = np.random.uniform(0.3, 1.5, 5)
    adv = np.random.rand(5)
    dist_entropy = 0.3
    epsilon = 0.4
    entropy_weight = 0.03

    policy_loss3 = policy_loss_fn(ratio, adv, dist_entropy, epsilon, entropy_weight)
    sol3 = -0.5512983844218763
    if np.allclose(policy_loss3, sol3):
        point += 1
    else:
        print(f"Expected {sol3}, but got {policy_loss3}")

    print(f"Policy loss total points: {point}/3")


def check_value_loss(value_loss_fn):
    import numpy as np
    point = 0
    np.random.seed(0)
    values = np.random.rand(5)
    returns = np.random.rand(5)
    value_loss1 = value_loss_fn(values, returns)
    if value_loss1 is None:
        raise NotImplementedError
    sol1 = 0.06940152136637252

    if np.allclose(value_loss1, sol1):
        point += 1
    else:
        print(f"Expected {sol1}, but got {value_loss1}")

    # test 2
    values = np.random.rand(5)
    returns = np.random.rand(5)
    value_loss2 = value_loss_fn(values, returns)
    sol2 = 0.2970616705780637
    if np.allclose(value_loss2, sol2):
        point += 1
    else:
        print(f"Expected {sol2}, but got {value_loss2}")


    # test 3
    values = np.random.rand(5)
    returns = np.random.rand(5)
    value_loss3 = value_loss_fn(values, returns)
    sol3 = 0.18660598503452355
    if np.allclose(value_loss3, sol3):
        point += 1
    else:
        print(f"Expected {sol3}, but got {value_loss3}")

    print(f"Value loss total points: {point}/3")
