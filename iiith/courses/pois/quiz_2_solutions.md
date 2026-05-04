# Quiz 2 Solution

I should have made this writeup earlier but now I am just trying to speedrun all solutions quickly.

### Problem 1

This is a variant of El-Gamal encryption

To revise,

ElGamal encryption scheme is a public-key cryptosystem

Based on the difficulty of solving the Discrete Logarithm Problem

ElGamal uses a pair of keys:
1. Public key → shared with everyone
2. Private key → kept secret

Its working,

1. Step 1: Key generation

We choose a large prime number p.

Then we chooose a generator g.

Then we pick a private key x (which is a random number)

And then we compute the public value

h = g^x (mod p)

Public key = (p, g, h)
Private key = x

2. Step 2: Encryption

To encrypt a message m:

a. We first choose a random number k.   
b. Then we compute:

c1 = g^k mod p

c2 = m * h^k mod p

Ciphertext = (c1, c2)

3. Decryption:

Using private key x:

a. Compute the shared secret

    s = c1^x mod p

    m = c2 * s^-1 mod p 


Coming back to the question

here they did c2 = h^r + m mod p


To prove it is not CPA-secure, we can construct an adversary $$\mathcal{A}$$ that wins the indistinguishability (IND-CPA) game with a non-negligible advantage.

Challender generates the public key. A chooses two distinct messages m0 = 0 and m1 = 1 and sends them to the challenger.

The challenge is when the challenger randomly selects a bit which belongs to {0, 1}.

The challenger the computes the ciphertext

and sends (c1, c2) back to A.

A must guess the bit b. Knowing m0 = 0. Then do something with quad residue, then the prob stuff and show that prob to win is 3 /4 and the advantage is 1/4.

### Problem 2

Raise s to the e,

Extract the message

Check last l bits of the message

The scheme is insecure because you can take any arbitary value and take the last l bits of it and get the message.

In a secure signature scheme, an attacker should not be able to produce a valid signature for any message without the private key.

### Problem 3

For a PRG to be considered secure, its output must be computationally indistinguishable from a truly random sequence of the same length

This means no efficient (polynomial-time) adversary or algorithm should be able to look at the output and tell whether it came from the PRG or from a truly random source.

The DDH assumption is a foundational concept in cryptography. It states that in certain cyclic groups, if you are given a generator $g$ and the values $$g^\alpha$$ and $$g^\beta$$ (where $$\alpha$$ and $$\beta$$ are random and secret), it is computationally infeasible to distinguish the value $$g^{\alpha\beta}$$ from a completely random group element $$g^\gamma$$.

Just apply DDH assumption and this is done.

### Problem 4

Chinese Remainder Theorem

It essentially tells us that if we know the remainders of an unknown number when divided by several different integers (that don't share any factors), we can uniquely determine that number within a certain range.

I'll see a video on this since I would not remember it otherwise

### Problem 5

Euler totient function, chines remainder theorem, two cases p and q.

### Problem 6

In RSA,

choose e such that gcd(e, phi(n)) = 1

Then compute private exponent d

d is the modular inverse of e mod phi(n)

d * e = 1 (mod (phi(n)))

d = e^-1 mod(phi(n))

### Problem 7

It is deterministic.

In a public-key setting, any deterministic encryption scheme trivially fails to achieve Chosen-Plaintext Attack (CPA) security.

Adversary can also do its own encryption and compare it with the challenger's encryption to see which b was chosen.

### Problem 8

We must demonstrate that an adversary can use a decryption oracle to gain information about a challenge ciphertext.

The core reason ElGamal fails CCA security is its malleability. Specifically, if you have a ciphertext for a message $$m$$, you can easily transform it into a ciphertext for a related message (like $2m$) without knowing the private key.

Have an s

(c1*, s x c2*)

And now the decryption oracle has to decrypt.

### Problem 9

Easy

### Problem 10

Skipping Miller Rabbin Test

### Problem 11

### Problem 13

Shamir's scheme

ye wala vahi independence probability show karna hai

### Problem 14

Too hard to understand rn, moved ahead.

### Problem 15

This is standard polynomial + Lagrange interpolation question.

### Problem 16

