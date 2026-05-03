---
layout: default
title: Endsem Solutions - POIS
---
# POIS Endsem Solutions

Quantum Mechanics does freak me out at times. I do admit messing up the QM part in Science-1 and it took me way long to get these things through science courses.

But these problems were the first time I found QM to get me hooked to how cool applications it has.

I would start with the first section:

## Topic 3: Basics of QM

#### Problem 8
While it says basics of QM, it certainly bombards you with seemingly complex ideas of Hilbert Spaces. I am not very good at definitions but I found that Hilbert space is just Euclidean space extended to many dimensions.

So instead of (x, y, z), you have more (x, y, z, a, b, c, d, A, n, s, h, C, ...)

And just this knowledge makes Problem 8 very simple and smooth like Amul Butter 😋

Vectors are linearly dependent if somehow you can add them in a way that makes them zero. If you can, then yes they are linearly dependent.

Let's see what magic can I cook (can you cook magic?)

We have P1 = (1, -1)

P2 = (1, 2)

and P3 = (2, 1)

So, let's assume for some `a`, `b`, `c`,

a*P1 + b*P2 + c*P3 = 0


=>

$$a(1,-1) + b(1,2) + c(2,1) = (0,0)$$

$$(a + b + 2c,\; -a + 2b + c) = (0,0)$$

So we get the system:

$$a + b + 2c = 0$$

$$-a + 2b + c = 0$$

Add the equations:

$$3b + 3c = 0 \Rightarrow b = -c$$

Substitute back:

$$a + (-c) + 2c = 0 \Rightarrow a + c = 0 \Rightarrow a = -c$$

So:

$$a = -c,\quad b = -c$$

Pick $$ c = 1 $$:

$$a = -1,\quad b = -1,\quad c = 1$$

Check:

$$-1(1,-1) + -1(1,2) + 1(2,1) = (0,0)$$


We do get 0! Which means they are linearly dependent.

<hr>

#### Problem 9

That was pretty easy. Now we get into something more complex. Who is the Pauli?

I looked up and the guy looked like this:

When young:
<img src = "images/pauli_young.png" width = 100>

But then he lost some of his hair:
<img src = "images/pauli_old.png" width = 100>

Now, I wouldn't really go to the solution directly but will first try to understand the question myself one by one.

The Pauli Matrices are given by:

$$
X =
\begin{pmatrix}
0 & 1 \\
1 & 0
\end{pmatrix}
$$

$$
Y =
\begin{pmatrix}
0 & -i \\
i & 0
\end{pmatrix}
$$

$$
Z =
\begin{pmatrix}
1 & 0 \\
0 & -1
\end{pmatrix}
$$

For now, I will take it as a fact that they look like this.

and it is says something which I cannot understand. It says that these can be considered operators with respect to the orthonormal basis $\vert 0\rangle$, $\vert 1\rangle$ for a 2D Hilbert Space.

It said too many things. I will try to break it down.

First I do understand the Hilbert Space (looked at it when seeing Problem 8 - that it is just your normal Euclidean space extended to more dimensions)

So so so,

Now what is orthonormal basis? Don't think it has to do anything with Quantum. Orthonormal is just ortho + normal.

Ortho means bones... no wait 😂.. ortho means they are orthogonal (perpendicular) to each other.

But what are perpendicular to each other?

$\vert 0\rangle$ and $\vert 1\rangle$.

$\vert 0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$ and $\vert 1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$


So yes these are orthogonal. Which we can easily see if we take the dot product.
1i + 0j \cdot 0*i + 1*j = 0


Now what does, "can be considered as operators" mean? It simply means that you can multiply it with other things.

You can do $X \vert 0\rangle$, etc. Just that.

Now we have to express each of these "Pauli operators" in the outer-product notation.

Outer product looks like $\vert a\rangle\langle b\vert $. It is when two basis vectors fall in love.

You can have four cases:

$$ \vert 0\rangle\langle 0\vert  = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \begin{pmatrix} 1 & 0 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} $$
$$ \vert 0\rangle\langle 1\vert  = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \begin{pmatrix} 0 & 1 \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix} $$
$$ \vert 1\rangle\langle 0\vert  = \begin{pmatrix} 0 \\ 1 \end{pmatrix} \begin{pmatrix} 1 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix} $$
$$ \vert 1\rangle\langle 1\vert  = \begin{pmatrix} 0 \\ 1 \end{pmatrix} \begin{pmatrix} 0 & 1 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix} $$

If we have these, then we can just add these get get or Pauli Matrices like follows:

$$
X = \lvert 0\rangle\langle 1\rvert + \lvert 1\rangle\langle 0\rvert
$$

$$
Z = \lvert 0\rangle\langle 0\rvert - \lvert 1\rangle\langle 1\rvert
$$

$$
Y = i\lvert 1\rangle\langle 0\rvert - i\lvert 0\rangle\langle 1\rvert
$$

Easy peasy sneezy

<hr>

#### Problem 10

That was another easy one, now onto Problem 10.

I would really just ignore the Medium and Hard tags and I think it was really stupid of the TAs to put it and intimidate students about the perceived difficulty. People go on solving unsolved problems if no one told them they were unsolved.

Anyways, Eigendecomposition - too big of a word but of Pauli Matrices which we know about.

Let me search for the word Eigendecomposition:

So basically we do something that lets us get eigenvectors and eigenvalues.

I have now also forgotten much of eigenvectors and eigenvalues but let's see how much can I do.

According to the question, we need to find the eigenvectors, eigenvalues and something else - diagonal (spectral) represntations (which I will ignore for now and find later)

of Pauli Matrices X, Y, Z.

This is what I will do first and then see what the other part of the question says.

To find the eigen value we just do

$$
\det(A - \lambda I) = 0
$$

(just remember this!)

And doing this, we get,

For $$Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$:

$$
\det \begin{pmatrix} 1 - \lambda & 0 \\ 0 & -1 - \lambda \end{pmatrix}
= (1 - \lambda)(-1 - \lambda) = 0
$$

$$
\lambda = \pm 1
$$

For $$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$:

$$
\det \begin{pmatrix} -\lambda & 1 \\ 1 & -\lambda \end{pmatrix}
= \lambda^2 - 1 = 0
$$

$$
\lambda = \pm 1
$$

For $$Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$$:

$$
\det \begin{pmatrix} -\lambda & -i \\ i & -\lambda \end{pmatrix}
= \lambda^2 - 1 = 0
$$

$$
\lambda = \pm 1
$$



---

Now to find eigenvectors, we have to follow the following procedure

$$
(A - \lambda I)v = 0
$$

Take $$v = \begin{pmatrix} x \\ y \end{pmatrix}$$ and solve explicitly.

---

And doing this would lead us to

For $$Z$$:

$$
\lambda = 1 \Rightarrow (Z - I)v =
\begin{pmatrix} 0 & 0 \\ 0 & -2 \end{pmatrix}
\begin{pmatrix} x \\ y \end{pmatrix} =
\begin{pmatrix} 0 \\ -2y \end{pmatrix} = 0
$$

$$
-2y = 0 \Rightarrow y = 0
$$

$$
v = \begin{pmatrix} x \\ 0 \end{pmatrix} \Rightarrow \text{choose } x=1
\Rightarrow v = \begin{pmatrix} 1 \\ 0 \end{pmatrix}
$$

---

$$
\lambda = -1 \Rightarrow (Z + I)v =
\begin{pmatrix} 2 & 0 \\ 0 & 0 \end{pmatrix}
\begin{pmatrix} x \\ y \end{pmatrix} =
\begin{pmatrix} 2x \\ 0 \end{pmatrix} = 0
$$

$$
2x = 0 \Rightarrow x = 0
$$

$$
v = \begin{pmatrix} 0 \\ y \end{pmatrix} \Rightarrow \text{choose } y=1
\Rightarrow v = \begin{pmatrix} 0 \\ 1 \end{pmatrix}
$$

---

For $$X$$:

$$
\lambda = 1 \Rightarrow (X - I)v =
\begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}
\begin{pmatrix} x \\ y \end{pmatrix} =
\begin{pmatrix} -x + y \\ x - y \end{pmatrix} = 0
$$

$$
-x + y = 0 \Rightarrow y = x
$$

$$
v = \begin{pmatrix} x \\ x \end{pmatrix} \Rightarrow \text{choose } x=1
\Rightarrow v = \begin{pmatrix} 1 \\ 1 \end{pmatrix}
$$

Normalize:

$$
\frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix}
$$

---

$$
\lambda = -1 \Rightarrow (X + I)v =
\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}
\begin{pmatrix} x \\ y \end{pmatrix} =
\begin{pmatrix} x + y \\ x + y \end{pmatrix} = 0
$$

$$
x + y = 0 \Rightarrow y = -x
$$

$$
v = \begin{pmatrix} x \\ -x \end{pmatrix} \Rightarrow \text{choose } x=1
\Rightarrow v = \begin{pmatrix} 1 \\ -1 \end{pmatrix}
$$

Normalize:

$$
\frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ -1 \end{pmatrix}
$$

---

For $$Y$$:

$$
\lambda = 1 \Rightarrow (Y - I)v =
\begin{pmatrix} -1 & -i \\ i & -1 \end{pmatrix}
\begin{pmatrix} x \\ y \end{pmatrix} =
\begin{pmatrix} -x - iy \\ ix - y \end{pmatrix} = 0
$$

From first equation:

$$
-x - iy = 0 \Rightarrow x = -iy
$$

Substitute into second:

$$
i(-iy) - y = (-i^2)y - y = (1)y - y = 0
$$

$$
v = \begin{pmatrix} -iy \\ y \end{pmatrix} \Rightarrow \text{choose } y=1
\Rightarrow v = \begin{pmatrix} -i \\ 1 \end{pmatrix}
$$

Normalize:

$$
\frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ i \end{pmatrix}
$$

---

$$
\lambda = -1 \Rightarrow (Y + I)v =
\begin{pmatrix} 1 & -i \\ i & 1 \end{pmatrix}
\begin{pmatrix} x \\ y \end{pmatrix} =
\begin{pmatrix} x - iy \\ ix + y \end{pmatrix} = 0
$$

$$
x - iy = 0 \Rightarrow x = iy
$$

Substitute:

$$
i(iy) + y = (-y) + y = 0
$$

$$
v = \begin{pmatrix} iy \\ y \end{pmatrix} \Rightarrow \text{choose } y=1
\Rightarrow v = \begin{pmatrix} i \\ 1 \end{pmatrix}
$$

Normalize:

$$
\frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ -i \end{pmatrix}
$$

---

Now about the weird diagonal (spectral representation)

Start with outer product:

$$\vert v\rangle \langle v\vert$$

Example for $$\vert+\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix}1 \\ 1\end{pmatrix}$$:

$$
\vert+\rangle \langle +\vert =
\frac{1}{2}
\begin{pmatrix}
1 \\ 1
\end{pmatrix}
\begin{pmatrix}
1 & 1
\end{pmatrix}
=
\frac{1}{2}
\begin{pmatrix}
1 & 1 \\
1 & 1
\end{pmatrix}
$$

Same for $$\vert-\rangle$$:

$$
\vert-\rangle \langle -\vert =
\frac{1}{2}
\begin{pmatrix}
1 & -1 \\
-1 & 1
\end{pmatrix}
$$

Now combine using eigenvalues:

$$
X = (+1)\vert+\rangle \langle +\vert + (-1)\vert-\rangle \langle -\vert 
$$

$$
X =
\frac{1}{2}
\begin{pmatrix}
1 & 1 \\
1 & 1
\end{pmatrix}
-
\frac{1}{2}
\begin{pmatrix}
1 & -1 \\
-1 & 1
\end{pmatrix}
=
\begin{pmatrix}
0 & 1 \\
1 & 0
\end{pmatrix}
$$

Same idea for all:

$$
Z = \vert0\rangle \langle 0\vert - \vert1\rangle \langle 1\vert
$$

$$
Y = \vert+i\rangle \langle +i\vert - \vert-i\rangle \langle -i\vert
$$

Tbh, there was nothing "medium" about it!! This was easy if you look at it, just a lot of work to do.

<hr>

#### Problem 11

Onto Problem 11 (my birthday is also on 11th!!! of another month 😲)

Now achanak se these people have started to say words like Projector

![Projector](https://64.media.tumblr.com/38dd33d5a4db185fe4928de5529da18d/4e175cff6f8d661a-50/s1280x1920/21e36f04c49f6be62e6daf5df940c79600627ecd.png)

How do I convince people that some projectors (especially the one in the image above does not follow $$P^2 = P$$ obviously)

![All jokes aside](https://media.tenor.com/GIZp7zXk8WYAAAAe/all-jokes-aside-rickey.png)

What is a projector? Well it is just a matrix which when multiplied to another matrix takes it to a different space.

For example:

[put the keep the 0 part and removes the one part example here]

Now the (a) part

Intutively if we think about it

First time applying P projects onto the direction,

And since we are already in that direction, multiplying it again would not change anything. So it is equivanlent to applying it once or twice or any number of times.

Let's see,

If we are applying a projection on \vert a> hoping to get a component of it along v, we do:

P∣a⟩=(something along ∣v⟩)

If the result is along v, it must look like:

(some number)×∣v⟩

P∣a⟩=c∣v⟩

What should c be?

From linear algebra, we know that is has to be related to the dot product, when we are trying to find the component along a direction.

Dot product in quantum forms is written. (please see this video if if not clear: https://www.youtube.com/watch?v=3N2vN76E-QA&t=98s)

c=⟨v∣a⟩

So the projection would be:

P∣a⟩=(⟨v∣a⟩)∣v⟩

Did you get this part? I did but I am not sure if it is clear here.


Now this is some mathematical (easy) derivation here but not relevant to the question, so I'll skip it.

At the end of the derivation, you eventually end up getting

P=∣v⟩⟨v∣

See it works->

(∣v⟩⟨v∣)∣ψ⟩=∣v⟩(⟨v∣ψ⟩)


Solving the actual (a) part problem:

P2=(∣v⟩⟨v∣)(∣v⟩⟨v∣)

Group it,

=∣v⟩(⟨v∣v⟩)⟨v∣

(we know that ⟨v∣v⟩=1) (why? For now just remember)

So, P2=∣v⟩⟨v∣=P

Kitna easy peasy tha.

Ab (b) We have to show that a normal matrix is Hermitian if and only if it has real eigenvalues which means it lives on the planet Hermi (get it? Martian?)

`if and only if` means that we have to do two things:

1. Show that a normal matrix has real eigenvalues, it is Hermitian

2. Show that a Normal + Hermitian matrix has real eigenvalues


What is Hermitian now?

So numbers can be complex and complex numbers can be put into matrices

Complex numbers can have conjugates, which is just replacing i with -i. So a + bi becomes a - bi, etc.

If we take the conjugate of all the numbers in a matrix and then take its transpose, we would have a matrix's Hermitian. hahahhahahahahaha

$$ a + bi \xrightarrow{\text{Conjugate}} a - bi \xrightarrow{\text{Transpose}} \text{Hermitian Conjugate (} \dagger \text{)} $$

Part 1 (normal + real eigenvalues -> Hermitian)

Fact: Normal matrices can be diagonalised 


If eigenvalues are real D^! = D

$$ A = UDU^\dagger $$
$$ A^\dagger = (UDU^\dagger)^\dagger = U D^\dagger U^\dagger $$
$$ D^\dagger = D \implies A^\dagger = UDU^\dagger = A $$



Part 2 (Hermitian -> Real eigenvalues)

$$ A\vert v\rangle = \lambda\vert v\rangle$$

Multiply $\langle v\vert$ both sides. Why? Because it gets you to the answer and it works you don't really have much time before the exam to understand how did people discover it.

$$ \langle v\vert A\vert v\rangle = \lambda\langle v\vert v\rangle $$
$$ (\langle v\vert A\vert v\rangle)^* = \langle v\vert A^\dagger\vert v\rangle = \langle v\vert A\vert v\rangle $$
$$ \lambda^* \langle v\vert v\rangle = \lambda\langle v\vert v\rangle \implies \lambda^* = \lambda $$


Now (c) part. We have to show that U xor V is unitary.

Unitary ka simple sa matlab hota hai U^†U=I

$$ (U \otimes V)^\dagger (U \otimes V) = (U^\dagger \otimes V^\dagger) (U \otimes V) $$
$$ = (U^\dagger U) \otimes (V^\dagger V) = I \otimes I = I $$

Ho gaye basics!! 


## Topic 4: BB84 Quantum Key Distribution

Now whenever I read BB84, I mistake it for the cool BB8 robot. I once tried to make a life-sized version of this cute robot in RRC (along with a life-sized version of Wall-E) but didn't have time and got busy with something(/someone) else.

Lite, will make these in my 5th year.

![BB-8](https://i.guim.co.uk/img/media/977085bbc667cb2819958790264d2d4fdb6ca689/0_0_2100_1260/master/2100.jpg?width=1200&height=1200&quality=85&auto=format&fit=crop&s=65c3c752b6c117a6525081b3007b8892)


Back to BB84

Now why is it called BB84 is because Charles Bennett and Gilles Brassard proposed it in 1984. See two Bs in the surnames.

Honestly this looks complex and I will go slow with it but I do like to think that things aren't that complex and I would eventually figure it out.

I read about it and the idea is very simple.

I am honestly tired with Alice and Bob, I'll use Indian names.

Khushi sends a message to Shashank.

She can send in two basis:

### Basis 1 (Z basis)

0 -> $\vert 0\rangle$

1 -> $\vert 1\rangle$


### Basis 1 (X basis)

0 -> $\vert+45^\circ\rangle$

1 -> $\vert-45^\circ\rangle$


Now Khushi wants to keep it secret about whatever she wants to say to Shashank. She can obviously go to him and wisper in his ear, but no, she wants to complicate things and make us understand BB84. (how bad jokes are these 🤦‍♂️)

Now she says something in two ways - either she can say it normally (Basis 1) or in a sarcastic tone (Basis 2) and now Shashank (who is a clear example that boys struggle in relationships) has to guess whether she said it normally or sarcastically. Shashank is clueless and he randomly chooses one Basis.

Now if he chooses the same Basis, he would get the same result.

If she said it normally (Basis 1) and he assumed that she said it normally (Basis 1), then life is set. And similary is she said it sarcastically (Basis 2) and he assumes (Basis 2) then life is set again.

The problem is when she says things sarcatically and he hears then normally or vice versa.

Itna to easy hai.

This is literally Problem 12.

I will try to convert everything into math now;

- Basis 1 (normal tone) → Z basis → $$\vert 0\rangle, \vert 1\rangle$$
- Basis 2 (sarcastic tone) → X basis → $$\vert +\rangle, \vert -\rangle$$

(a) If Shashank chooses the wrong basis, the result is random

Suppose she sends:
- $$\vert 0\rangle$$ (normal tone)

But Shashank assumes she is saying in sarcastic tone:
- $$\vert +\rangle, \vert -\rangle$$

Now compute:

$$
\vert +\rangle = \frac{\vert 0\rangle + \vert 1\rangle}{\sqrt{2}}, \quad
\vert -\rangle = \frac{\vert 0\rangle - \vert 1\rangle}{\sqrt{2}}
$$

Probability of getting $$\vert +\rangle$$

$$
\vert \langle + \vert  0 \rangle\vert ^2
= \left\vert \frac{1}{\sqrt{2}}(\langle 0\vert  + \langle 1\vert )\vert 0\rangle\right\vert ^2
= \left\vert \frac{1}{\sqrt{2}}\right\vert ^2
= \frac{1}{2}
$$

---

## Probability of getting $$\vert -\rangle$$

$$
\vert \langle - \vert  0 \rangle\vert ^2 = \frac{1}{2}
$$

---

## Conclusion

When basis is different:
- outcomes are $$1/2, 1/2$$
- completely random
- independent of what she actually sent

This proves part (a)

---

# (b) If Shashank chooses the **correct basis**, result is perfect

Case 1:
- She sends $$\vert 0\rangle$$
- He measures in Z basis

$$
\vert \langle 0 \vert  0 \rangle\vert ^2 = 1
$$

So he gets the correct answer with probability 1.

---

Case 2:
- She sends $$\vert +\rangle$$
- He measures in X basis

$$
\vert \langle + \vert  + \rangle\vert ^2 = 1
$$

Again, perfect match.

---

## Conclusion

When basis matches:
- outcome is deterministic
- Shashank gets exactly what she meant

This proves part (b)

<hr>

#### Problem 13

Problem 13 looks scary to me. Itna kuch likha hai, samjhna mushil hai.

But very honestly anyone who knows counting can solve it.

The big picture really is:

1. There are 2n total bits. Khushi ko itna bolna hai. Kitna bolte hain log. Chup hi nahi hote.
2. Total errors = μn, for some μ.
3. We can randomy pick n check bits.
4. The other n bits are untested.


We literally just have to show that if the bits we are chicken - did I chicken, yeah so the bits we are checking, if they look good, then it is very less likely that the remaining bits are bad.

Basically Khushi ki 2n baaton mein se koi random n baatein check kar rahe hain aur agar wo theek lagi to bohot kam chance hai ki baki baatein theek nahi hongi.

Mathematically, the probability is very low and shrinks exponentially like

$$ \exp(-\text{constant}\cdot \epsilon^2 n) $$

Read the question 2-3 times and you would really understand that the question is asking this exact thing. Profs have an ability to scare people and not teach them properly, bruh.

Now Khushi said 2n things and it could have μn errors (0 to 2n which mean μ can go from 0 to 2)

Now we are not checking everything. We are just checking n bits and out of those n bits if there are δn errors and (δ + ϵ)n on the rest. Then we have to show that δ = (µ − ϵ)/2.

Tell me which 6th grader would not be able to do it.

Total errors = μn
Errors in those checked = δn
Errors in those not checked = (δ + ϵ)n

So δn + (δ + ϵ)n = μn

=> (2δ + ϵ)n = μn

=> δ = (µ − ϵ)/2

Now we have to use this to show that the probability p satisfies,

We are picking n check bits out of 2n check bits

$$ \binom{2n}{n} $$

We want exactly δn errors in check bits

remaining errors go into the other n bits

$$ \binom{\mu n}{\delta n} \binom{2n - \mu n}{n - \delta n} $$

$$ p = \frac{\binom{\mu n}{\delta n} \binom{(2 - \mu)n}{(1 - \delta)n}}{\binom{2n}{n}} $$


(2) is also very easy now



I would stop here for a while and reflect upon the abobve solution. I was really scared seeing this but now it is very very clear to me.


Onto Problem 14

Problem 14 is also what we see in daily life. There is another girl Arushi who wants to read Khushi's messages without getting caught.

Khushi sends messages using two styles.

- Basis 1 (normal tone):  
  0 → "main theek hun"  
  1 → "haan chalega mujhe"

- Basis 2 (sarcastic tone):  
  0 → "main theek hun 😒"  
  1 → "haan chalega mujhe 😒"

So total possible messages:
- "haan chalega mujhe"  
- "main theek hun"  
- "haan chalega mujhe 😒"  
- "main theek hun 😒"

These correspond to:
$$
|0\rangle,\ |1\rangle,\ |+\rangle,\ |-\rangle
$$

Arushi tries to listen in.

But the problem is that even she does not know what is the real tone with with khooshi is speaking.

So if she guesses wrong, she misunderstands, sends wrong info to Shashank and gets caught 💀 - uske baad to situation sahi nahi rahegi.

Now assume Arushi has jadui shakti. She can perfectly tell what Khushi meant even if tone is unclear.

This means she can distinguish all 4 states perfectly even though they are not orthogonal.

(put rigourous mathematical explanation here)

## Topic 5: Quantum Teleportation

![Phineas and Ferb Teleporter](https://media.tenor.com/83pA8Vq7lAIAAAAd/phineas-and-ferb-teleport.gif)

(please watch this video: https://www.youtube.com/watch?v=R8gbj-X1p9w - great fun)

#### Problem 15

Let's forget about Quantum Teleportation for this question since it is not being used here. It is just building the basis.

Let's get a bit into Quantum Concepts. I'll keep a bit serious tone here to explain things properly. Pay close attention.

1. What is a bit (normal computing)

It is just 0 or 1!

2. What is Qubit

It is also a bit (Qu bit) but it can be a mix of both 0 and 1 at the same time. Like a linear combination.

$\vert \psi\rangle = a\vert 0\rangle + b\vert 1\rangle$

$\vert 0\rangle$ = state 0
$\vert 1\rangle$ = state 1

a, b are just some general numbers to tell how a Qubit is formed.

3. Measurement

I do not know how many sci-fi movies have you guys seen but I clearly know from somewhere (I don't know where - maybe Papa told me once) that when we measure a Qubit, it becomes either 0 or 1.

Just measuring it like a chuimui, it changes to either 0 or 1 from whatever combination or blend it was in before.

And probabilities depend on the values a and b as we had before.

Now we will see Quantum Gates

Just like the normal gates

Example: NOT gate (X)

Flips:

$\vert 0\rangle \to \vert 1\rangle$
$\vert 1\rangle \to \vert 0\rangle$

Example: Hadamard (H)

Creates superposition:

$\vert 0\rangle \to \frac{\vert 0\rangle+\vert 1\rangle}{\sqrt{2}}$

Since time is less, I would just say superposition is sum of two states - which is what it really is.

Now, what is a two-qubit system?

If we combine two-qubits, we get a two-qubits system - simpi simpa.

So, $\vert 00\rangle$, $\vert 01\rangle$, $\vert 10\rangle$, $\vert 11\rangle$

These are like all combinations of two bits.

Entanglement (and EPF paradox) can be very easily understood through this video (https://youtu.be/5HJK5tQIT4A?si=LGePtfQKJEQLSgYO)

Without going into much physical significance, let's solve Q15.

A Bell State looks like

$\frac{\vert 00\rangle+\vert 11\rangle}{\sqrt{2}}$

(this is entangled and not directly readable)

What we really want is to figure out which Bell state we have (out of the four) and our measurement only understands $\vert 00\rangle$, $\vert 01\rangle$, $\vert 10\rangle$, $\vert 11\rangle$.

So we will do what the question tells us to do:

$U = \text{CNOT} \cdot (H \otimes I)$

CNOT does:

$\vert 00\rangle \to \vert 00\rangle$
$\vert 11\rangle \to \vert 10\rangle$

$$ \frac{\vert 00\rangle+\vert 11\rangle}{\sqrt{2}} \xrightarrow{\text{CNOT}} \frac{\vert 00\rangle+\vert 10\rangle}{\sqrt{2}} $$

Apply H to first Qubit and we are done.

$$ \frac{\vert 0\rangle+\vert 1\rangle}{\sqrt{2}} \otimes \vert 0\rangle = \vert +\rangle \otimes \vert 0\rangle \xrightarrow{H \otimes I} \vert 0\rangle \otimes \vert 0\rangle = \vert 00\rangle $$


Measurement operators (what are they?)


I think we are getting a hand of this now.


#### Problem 16

Ignore the title, let's me see what this means. We have a composite system (which means more one one thing is there). $\vert a\rangle$ is a pure state of A and $\vert b\rangle$ is a pure state of B.

What is a pure state?

Pure state just means ki we know the quantum state completely. There are cases when you don't know the quantum state completely. This video is a good explanation: [Video Link](https://youtu.be/gFtt0C4enZA?si=744OMw6O_1nOjr5I)

A pure state can also be written as:

$\rho = \vert \psi\rangle\langle\psi\vert $

just another way to represent the same thing.

For pure states:

$\rho^2 = \rho$

Pavitr cheez mein Pavitr cheez daalo to Pavitr hi hota hai? (Maybe I can remember this properly like this)

I think now we get what a pure state is.

Now I have to show something about the reduced density operator.

What is that now? 😭

Reduced density operator = “what A looks like if I don’t see B”

What my .... (give some funny analogy)


$$ \rho = \vert a\rangle\langle a\vert  \otimes \vert b\rangle\langle b\vert  \implies \rho_A = \text{Tr}_B(\rho) = \vert a\rangle\langle a\vert  \text{ (Pure)} $$
$$ \rho = \vert \Phi^+\rangle\langle\Phi^+\vert  \implies \rho_A = \frac{1}{2}(\vert 0\rangle\langle 0\vert  + \vert 1\rangle\langle 1\vert ) = \frac{1}{2}I \text{ (Mixed)} $$



Let's move on to the last question of Quantum Teleportation. Honestly I have seen no teleportation happening yet.

Let's see what does Problem 17 say.

So again the question does not clarify a lot of things and just assumes the reader to figure it out.

So, me, the reader has to figure this out.😑

I see U being mentioned. I just assume that it is some gate (some operation) we would be doing on a Qubit.

It says that we have two ways to run a circuit:

Way 1

1. Use Qubit 1 to tell whether U is applied to qubit 2
2. Then measure Qubit 1

Way 2 (measure first)
1. Measure Qubit 1
2. If result = 1, apply U to Qubit 2
3. If result = 0, do nothing.

And all they want us to do is to prove that they give the same result.

Like same result means that we would have the same probabilities and same final states.

Controlled-U just means that if one qubit (called control) is 1, apply U.

Starting with the general input:

$\vert \psi\rangle = \alpha\vert 0\rangle\vert \phi_0\rangle + \beta\vert 1\rangle\vert \phi_1\rangle$

Step 1: Apply Controlled U. So U would be applied where Qubit is $\vert 1\rangle$ (control is 1)

$\alpha\vert 0\rangle\vert \phi_0\rangle + \beta\vert 1\rangle U\vert \phi_1\rangle$

Step 2: Measure Qubit 1

Two possibilities (like we know na ki measuring a Qubit makes it collapse into one of the two states $\vert 0\rangle$ or $\vert 1\rangle$)

1. Outcome 0 -> State Becomes

$\vert 0\rangle\vert \phi_0\rangle$

2. Outcome 1 -> state becomes

$\vert 1\rangle U\vert \phi_1\rangle$


Way 2

Measure first

Then apply U, we get the same result.

And now we see deferred measurements ka real matlab. You can move measurements later in the circuit without changing results.

Par honestly ye bacchon wale questions jisme bas complex notation aur darane wali cheezein use hui hai wo hum college walon ko de rahe hai aur bol rahe hai Medium Question hai and revise, read notes, don't worry. Kyunn??? Itne asaan questions ke liye. Bhap!!

(put nice explanation of how this principle is used in Quantum Teleportation to justify moving Alice's measurements to the end of the protocol.)


## Topic 6: The No-Cloning Theorem

Firstly watch this video (very good video) and a lot of concepts would be clear:

https://www.youtube.com/watch?v=owPC60Ue0BE

I got lost after 3:25 minutes but at least I know something now, so let's do Problem 18

I had explained (in Problem 14) what Perfectly distinguishing non-orthogonal states means and how that is impossible. [Impossible Thing A]

And the video explains how cloning quantum states is impossible. [Impossible Thing B]

Problem 18 just wants us to show if Impossible Thing A can happen then Impossible Thing B can happen and vice versa.

Part 1: If you can distinguish, you can clone.


$$ U(\vert \psi\rangle \otimes \vert 0\rangle) = \vert \psi\rangle \otimes \vert \psi\rangle $$
$$ U(\vert \phi\rangle \otimes \vert 0\rangle) = \vert \phi\rangle \otimes \vert \phi\rangle $$
$$ \langle \psi\vert \phi\rangle \langle 0\vert 0\rangle = \langle \psi\vert \phi\rangle^2 \implies \langle \psi\vert \phi\rangle = \langle \psi\vert \phi\rangle^2 $$
$$ \implies \langle \psi\vert \phi\rangle \in \{0, 1\} $$

Problem 19

I think this Orthogonal thing is getting too out of hand and I need to really understand it properly now.




Problem 20

$$ \rho' = \frac{1}{2}\vert v\rangle\langle v\vert  + \frac{1}{2}\vert v^\perp\rangle\langle v^\perp\vert  = \frac{1}{2}(\vert v\rangle\langle v\vert  + \vert v^\perp\rangle\langle v^\perp\vert ) = \frac{1}{2}I = \rho $$

Problem 21

Again labelled as hard, this is a bherry bherry pheasy quesson

## Topic 7: The EPR Paradox

This video was great when I looked at it: https://www.youtube.com/watch?v=5HJK5tQIT4A

Let's do problem 22 now:


(write explanation here)


For remaining questions,


پُڄائي نٿو 😔

مان ٿوري دير ۾ ڪندس
