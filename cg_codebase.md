# Code Explanation for the CG Codebase

After being overwhelmed with massive code bases, we often try to find where the main function is in the codebase.

In this case, it is in the  `render.cpp` file.

## Part 1: Taking a brief overview

Here is the main function:
```cpp
int main(int argc, char **argv)
{
    if (argc != 3) {
        std::cerr << "Usage: ./render <scene_config> <out_path>";
        return 1;
    }
    Scene scene(argv[1]);

    Integrator rayTracer(scene);
    auto renderTime = rayTracer.render();

    std::cout << "Render Time: " << std::to_string(renderTime / 1000.f) << " ms" << std::endl;
    rayTracer.outputImage.save(argv[2]);

    return 0;
}
```

### Part 1.1: Overview of what's called Scene

We first see the line `Scene scene(argv[1]);`. Here `Scene` is a struct as follows:
```cpp
struct Scene {
    std::vector<Surface> surfaces;
    Camera camera;
    Vector2i imageResolution;

    Scene() {};
    Scene(std::string sceneDirectory, std::string sceneJson);
    Scene(std::string pathToJson);
    
    void parse(std::string sceneDirectory, nlohmann::json sceneConfig);

    Interaction rayIntersect(Ray& ray);
};
```

- First, this `struct Scene` has a vector of structs `Surface` named surfaces.
- Then a `struct Camera` named camera.
- Then a 2D vector Vector2\<int> that has been renamed to Vector2i using `typedef Vector2<int> Vector2i;` and named imageResolution.

After this we see a bunch of constructors.
- The first one is an empty constructor. `Scene() {};`
- The second one has two parameters. `Scene(std::string sceneDirectory, std::string sceneJson);`
- The thirs one has only one. `Scene(std::string pathToJson);`
- As we have called `Scene scene(argv[1]);` <- One parameter, the last constructor function would be used.

<b>Note: On a careful observation you might note that the last 2 constructors are one and the same thing. </b> 

Now let's understand how this scene constructor works.
```cpp
1 Scene::Scene(std::string pathToJson)
2 {
3    std::string sceneDirectory;
4 
5 #ifdef _WIN32
6     const size_t last_slash_idx = pathToJson.rfind('\\');
7 #else
8     const size_t last_slash_idx = pathToJson.rfind('/');
9 #endif
10 
11    if (std::string::npos != last_slash_idx) {
12       sceneDirectory = pathToJson.substr(0, last_slash_idx);
13    }
14
15    nlohmann::json sceneConfig;
16    try {
17        std::ifstream sceneStream(pathToJson.c_str());
18        sceneStream >> sceneConfig;
19    }
20    catch (std::runtime_error e) {
21        std::cerr << "Could not load scene .json file." << std::endl;
22        exit(1);
23    }
24
25    this->parse(sceneDirectory, sceneConfig);
26 }
```

For most people, the code should be good until line 14. The rest of the lines is just more of knowing the functions than some damn hard coding.

We have imported an external git repo in the `extern/json` dir. That's where the `json` class (alias of `basic_json<>`) is being taken from the nlohmann namespace. This might make good sense to `C++11` coders. Others might just see this as something that contains `json` information.

The `parse()`` function takes the sceneConfig and populates the scene struct's camera and surfaces. The code for parse is rather straightforward when other aspects of code are understood.

Notice `this->parse(---)`. Here `this` refers to the `Scene struct` that was the first line we saw in the `main` function.

Now how the surfaces are created and what is the `Vector2i imageResolution` and more things, we will see later on in the following sections. ::Improve this line

### Part 1.2: Overview of what's called Integrator

Now our focus is on this line of code:
`Integrator rayTracer(scene);`

This is the next line of the main function.

Things are simple:
- `Integrator` is a struct.
- rayTracer is the name of this particular Integrator instance.
- We are passing a `scene` argument. This means there must be a constructor function somewhere.

```cpp
struct Integrator {
    Integrator(Scene& scene);

    long long render();

    Scene scene;
    Texture outputImage;
};
```

Ah, there it is. `Integrator(Scene& scene);` is the constructor function.

Here we make a few observations:
- The struct also has a Scene struct inside it.
- It also has a Texture struct inside it. The texture struct is named `outputImage`.

There is also this cute little `render()` function which is very long. <i>So long that `long` had to be written twice. </i>

<b> What does the Integrator function do? </b>

Well it takes a Scene (and does what?)

### Part 1.3: Calling render

The next part of the main function has `auto renderTime = rayTracer.render();`

After the Integrator does its job, we do rendering. This process has the code which does:
- Generate a ray
- Do ray intersection.
- Check if the ray intersected.
- Depending on whether or not it did, change the color.

There is also functionality to get the time taken for rendering.

### Part 1.4: Saving the image

After this the image is saved to a file. There are their own complex functions which we shall talk about later.

## Part 2: In-depth analysis of Camera Parsing

There was a time up in this explanation when we mentioned that the `parse` function populates the `Camera struct` and the vector `surfaces` of elements having type `struct Surface`.

We'll look at what it means to parse the values in the Camera struct and what is happening there along with the several terminologies.

Here's the code for Cameras: (in `scene.cpp`)
```cpp
// Cameras 
    try {
        auto cam = sceneConfig["camera"];

        this->camera = Camera(
            Vector3f(cam["from"][0], cam["from"][1], cam["from"][2]),
            Vector3f(cam["to"][0], cam["to"][1], cam["to"][2]),
            Vector3f(cam["up"][0], cam["up"][1], cam["up"][2]),
            float(cam["fieldOfView"]),
            this->imageResolution
        );
    }
    catch (nlohmann::json::exception e) {
        std::cerr << "No camera(s) defined. Atleast one camera should be defined." << std::endl;
        exit(1);
    }
```

More specifically this:
```cpp
auto cam = sceneConfig["camera"];

this->camera = Camera(
    Vector3f(cam["from"][0], cam["from"][1], cam["from"][2]),
    Vector3f(cam["to"][0], cam["to"][1], cam["to"][2]),
    Vector3f(cam["up"][0], cam["up"][1], cam["up"][2]),
    float(cam["fieldOfView"]),
    this->imageResolution
);
```

First let's look at `cam`.
Cam is the "camera" part of the sceneConfig.

Here is a sample sceneConfig:
```cpp
{
    "camera": {
        "fieldOfView": 30,
        "from": [0, -24, 5],
        "to": [0, 24, 5],
        "up": [0, 0, 1]
    },
    "output": {
        "resolution": [1080, 1080]
    },
    "surface": [
        "scene.obj"
    ]
}
```

The `auto` is used to automatically extract the complex type that camera has.

Now we see a function Camera. And it appears that `camera` inside the Scene struct is getting populated from this. <i>Hehe, it wouldn't be bad if you smile right now. Don't ask the reason please. We should be smiling forever until someone says that we are insane. Nvm, just a thought.</i>

It is very clear that we are extracting information here. Apart from the information that we can extract from the sceneConfig, we are also having a `this->imageResolution` being passed. We'll come to this a little later.

AND NOW!!! THE CAMERA FUNCTION.

Not much of a big deal. It is more of terminologies and simple math here:

See this and get afraid:
```**cpp**
Camera::Camera(Vector3f from, Vector3f to, Vector3f up, float fieldOfView, Vector2i imageResolution)
    : from(from),
    to(to),
    up(up),
    fieldOfView(fieldOfView),
    imageResolution(imageResolution)
{
    this->aspect = imageResolution.x / float(imageResolution.y);

    // Determine viewport dimensions in 3D
    float fovRadians = fieldOfView * M_PI / 180.f;
    float h = std::tan(fovRadians / 2.f);
    float viewportHeight = 2.f * h * this->focusDistance;
    float viewportWidth = viewportHeight * this->aspect;

    // Calculate basis vectors of the camera for the given transform
    this->w = Normalize(this->from - this->to);
    this->u = Normalize(Cross(up, this->w));
    this->v = Normalize(Cross(this->w, this->u));

    // Pixel delta vectors
    Vector3f viewportU = viewportWidth * this->u;
    Vector3f viewportV = viewportHeight * (-this->v);

    this->pixelDeltaU = viewportU / float(imageResolution.x);
    this->pixelDeltaV = viewportV / float(imageResolution.y);

    // Upper left
    this->upperLeft = from - this->w * this->focusDistance - viewportU / 2.f - viewportV / 2.f;
}
```

First we have
```cpp
Camera::Camera(Vector3f from, Vector3f to, Vector3f up, float fieldOfView, Vector2i imageResolution)
    : from(from),
    to(to),
    up(up),
    fieldOfView(fieldOfView),
    imageResolution(imageResolution)
```

The simplest explaination to this piece of code is:
This is the Camera struct
```cpp
struct Camera {
    Vector3f from, to, up;
    float fieldOfView;
    Vector2i imageResolution;

    float focusDistance = 1.f;
    float aspect;

    Vector3f u, v, w;
    Vector3f pixelDeltaU, pixelDeltaV;
    Vector3f upperLeft;

    Camera() {};
    Camera(Vector3f from, Vector3f to, Vector3f up, float fieldOfView, Vector2i imageResolution);

    Ray generateRay(int x, int y);
};
```

We have the Vector3f (alias of Vector3\<float>) that has the identifiers `from`, `to` and `up`. These three identifiers would be poulated using the code above directly without even entering the function.

The `from` to `to` makes a vector that specifies where the camera is looking at. And the `up` is the up vector as discussed in class by Prof PJN.

In the past, I had some problems understanding the up vector. Largely due to the different definitions it has. See the appendices for the explanation of this up vector.

<b>Field of View </b> (chappa from internet)

In the context of computer graphics and virtual cameras, the field of view (FOV) is an important parameter that defines the extent of the scene that is visible in the rendered image. It determines how much of the 3D world can be seen by the virtual camera and influences the perception of depth and perspective in the final rendered image.

<b>Image Resolution</b> (chappa from internet)

In computer graphics and rendering, image resolution refers to the number of pixels in an image, usually expressed as width and height dimensions.

So it makes sense to have the imageResolution as a Vector2i having 2 things to specify in integer format. <i>I honestly didn't like this. They should have made a separate typedef and called it the resHolder. I am often in favour of having multiple names of the same thing.</i>

These are the mathematical conversions. They should be clear if you think about them for a while. You can even think about them while brushing, bathing, etc instead of have those bad and worrying thoughts.
```cpp
// Determine viewport dimensions in 3D
float fovRadians = fieldOfView * M_PI / 180.f;
float h = std::tan(fovRadians / 2.f);
float viewportHeight = 2.f * h * this->focusDistance;
float viewportWidth = viewportHeight * this->aspect;
```

This is the simple calculation for the basis vectors
```cpp
// Calculate basis vectors of the camera for the given transform
this->w = Normalize(this->from - this->to);
this->u = Normalize(Cross(up, this->w));
this->v = Normalize(Cross(this->w, this->u));
```

Now comes an interesting part. Have a biscuit, if you like.

```cpp
// Pixel delta vectors
Vector3f viewportU = viewportWidth * this->u;
Vector3f viewportV = viewportHeight * (-this->v);

this->pixelDeltaU = viewportU / float(imageResolution.x);
this->pixelDeltaV = viewportV / float(imageResolution.y);
```

<b>P.I.X.E.L. D.E.L.T.A. V.E.C.T.O.R.S.</b>

Here's what they are:

Pixel delta vectors represent the change in the camera's coordinate system for each pixel in the rendered image. In other words, they define how much the camera's coordinate system changes as you move from one pixel to the next in the image. These vectors are used to determine the direction in which rays are cast from the camera for each pixel.

Think of them like this: we have an image that has a resolution and we have a viewport. Now we have to scale the image to the viewport. Doing this first asks for reducing to unit image resolution and then multiplying by the viewport dimensions.

For there calculation, we see that they would definitely depend on the viewportWidth and viewportHeight and would be in the direction of the basis vectors `u` and `v` (why?)

<b>Upper left corner</b> (chappa)

The upper-left corner of the viewport is a point in 3D space that corresponds to the top-left corner of the region being captured by the camera. This point is crucial for determining the starting position for casting rays into the scene.

And thus, we have successful fit the `Camera` information inside the `Scene`.

۽ اهڙيءَ طرح، اسان ڪاميابيءَ سان ’ڪئميرا‘ جي معلومات کي ’منظر‘ جي اندر درست ڪيو آهي

## Part 3: In-depth analysis of Surface Parsing

### Part 3.1: Introduction
Here is the code for surface parsing:
```cpp
auto surfacePaths = sceneConfig["surface"];

uint32_t surfaceIdx = 0;
for (std::string surfacePath : surfacePaths) {
    surfacePath = sceneDirectory + "/" + surfacePath;

    auto surf = createSurfaces(surfacePath, /*isLight=*/false, /*idx=*/surfaceIdx);
    this->surfaces.insert(this->surfaces.end(), surf.begin(), surf.end());

    surfaceIdx = surfaceIdx + surf.size();
}
```

If you came here after looking at Part 2, you may have no problems with the first line.

Again here is a sample sceneConfig:
```cpp
{
    "camera": {
        "fieldOfView": 30,
        "from": [0, -24, 5],
        "to": [0, 24, 5],
        "up": [0, 0, 1]
    },
    "output": {
        "resolution": [1080, 1080]
    },
    "surface": [
        "scene.obj"
    ]
}
```

The identifier `surfacePaths` then subsequently holds `"scene.obj"`. Note, there could be more scenes in the list as well. We are just considering one for now.

Umm, likely non C++ coders (even myself) might have some problems understanding the for loop above. Well just ask your friend in that case.

The surfacePath is first prefixed with the sceneDirectory to get the full path.

### Part 3.2: createSurfaces

createSurfaces turns out to be a good 125 lines of code.

This code defines a function createSurfaces that reads an OBJ file (Wavefront .obj format, representing 3D geometry) using the tinyobj library and converts it into a custom data structure called Surface. The Surface structure represents a 3D surface, storing information such as vertices, normals, UV coordinates, and material properties.

Here are the steps involved in doing so:
- An OBJ file is parsed. This OBJ file contains the elements for the 3D model. See `scenes/cornell_box/scene.obj` to find the file. You'll get an idea about how things are there.
- Shapes and Faces are iterated over.
- Surface Data Structure is built
  - For each face, Surface object is constructed, storing vertices, normals, UV coordinates, and material information.
- Textures are loaded. The function loads diffuse and alpha textures from material definitions if they are available.

<b>Diffuse Textures:</b>
It is the base color of an object without any lighting applied

<b>Alpha Textures:</b>
"Alpha" is the name given to the value that controls opacity/transparency - It doesn't strictly have a colour as such - it's sometimes displayed as being a greyscale texture

### Part 3.3 

After this we have:

`this->surfaces.insert(this->surfaces.end(), surf.begin(), surf.end());`

This line simply adds the surf obtained to the end of the vector `surfaces` in the `struct Scene`

This is how we increment the iterator: `surfaceIdx = surfaceIdx + surf.size();`

## Part 4: Rendering

### Part 4.1: Introduction

Ah, tis the part that most of us are excited about, like an electron. <i>I might soon release a form to rate my jokes.</i>

When we are done with initialising the integrator part, we call the render() function like this:

`auto renderTime = rayTracer.render();`

For now, we'll ignore the renderTime part.

We had previously made a struct `rayTracer` of the type `Integrator` and have initialised it with the `scene`. The scene's components were populated using the steps mentioned previously.

Now at the crux of it, here is the render function:
```cpp
for (int x = 0; x < this->scene.imageResolution.x; x++) {
        for (int y = 0; y < this->scene.imageResolution.y; y++) {
            Ray cameraRay = this->scene.camera.generateRay(x, y);
            Interaction si = this->scene.rayIntersect(cameraRay);

            if (si.didIntersect)
                this->outputImage.writePixelColor(0.5f * (si.n + Vector3f(1.f, 1.f, 1.f)), x, y);
            else
                this->outputImage.writePixelColor(Vector3f(0.f, 0.f, 0.f), x, y);
        }
    }
```

It goes as follows: (Important)

For each pixel, generate a ray from that pixel. Which pixel are we talking about, you may ask. Well, it is one of the pixels we have in our camera.

How many pixels are there. Well this depends on the `image resolution` as it is very clear from the two `for loops`.

Since `rayTracer` is an `Integrator`, `this.scene` refers to the `struct Scene` (capitalised) called `scene` (lowercase) inside the `Integrator struct`. You may want to see the `Integrator struct` again.

Then we see if that ray intersects with the scene.

Finally we would color the pixel accordingly.

Of course, they are ideally only two cases here: it would intersect or it would not.

- Well if it would, the color would be decided based on the surface normals. 
- If a ray does not intersect anything, then a black color is assigned to the pixel.

### Part 4.2: Generate Ray

This is the `generateRay` function:

```cpp
Ray Camera::generateRay(int x, int y)
{
    Vector3f pixelCenter = this->upperLeft + 0.5f * (this->pixelDeltaU + this->pixelDeltaV);
    pixelCenter = pixelCenter + x * this->pixelDeltaU + y * this->pixelDeltaV;

    Vector3f direction = Normalize(pixelCenter - this->from);

    return Ray(this->from, direction);
}
```

The pixel center is calculated from the upperLeft point (remember it? Just scroll up to find out) and the pixelDeltaU/V. This is simply to calulate the offset of the pixelCenter based on the camera. I understand it. If you don't, let me know I'll write more here about it here (or even make a video). For references, refer to the 3rd PPT for the course (in Spring '24).

`Vector3f direction = Normalize(pixelCenter - this->from);`

Then the direction of the ray is simple. (Include image here)

We have a from direction of the camera and we have the pixel position. Using this information we would be able to find out the direction of the ray.

Then by this: `return Ray(this->from, direction);`, a struct Ray is returned that is filled by the from point of the ray and its direction. These two things are sufficient to form a ray.

### Part 4.3: rayIntersect

When I was studying ray tracing some days back, I always wondered how do they find the intersections. 

Here is what `Billiman` has to tell you. <i>(Imagine billiman as a cat with a cape, standing on two legs, one hand forward and the second in attention)</i>

`Billiman` : Here is how rayIntersect is called:

`Interaction si = this->scene.rayIntersect(cameraRay);`

`Dumbo (you)`: What is `Interaction`?

`Billiman`: It is a struct:

```cpp
struct Interaction {
    Vector3f p, n;
    float t = 1e30f;
    bool didIntersect = false;
};
```

There are two Vector3fs (alias of Vector3\<float>) in here: p and n. 
- p is position. 
- n is normal.

Position is where the interaction happened. 

n is the normal of the surface at that point of interaction.

Notice a `t`. This  `t` is seemingly large and of type float. `(1e30)` means large and `f` for float.

Then there is information if the intersection did not happen using, `didIntersect` which is by default initialised to false.

`Dumbo (you)`: Why scene.rayIntersect(cameraRay)?

`Billiman` : It captures the intersection of the cameraRay with the scene.

`Less Dumbo (you)`: Ok, now tell me about the actual code.

`Billiman` : Here it is. It's only you for which I have taken this code out of my magical powers.

```cpp
Interaction Scene::rayIntersect(Ray& ray)
{
    Interaction siFinal;

    for (auto& surface : this->surfaces) {
        Interaction si = surface.rayIntersect(ray);
        if (si.t <= ray.t) {    
            siFinal = si;
            ray.t = si.t;
        }
    }

    return siFinal;
}
```

To find if a ray intersected with the scene, we would have to find if it intersected with a surface. Isn't that obvious? Yes, it is.

We'll go through all surfaces.

In more technical words.
- We have an Interaction `siFinal`. 
- We go over all the surfaces we have.
- For each surface, we have an Interaction `si` (surface interaction)
- This Interaction si is computed using another function of the same name `rayIntersect`. The functions can have same names as they belong to different structs.

The surface.rayIntersect(ray) is as follows:

(from surface.cpp)

```cpp
Interaction Surface::rayIntersect(Ray ray)
{
    Interaction siFinal;
    float tmin = ray.t;

    for (auto face : this->indices) {
        Vector3f p1 = this->vertices[face.x];
        Vector3f p2 = this->vertices[face.y];
        Vector3f p3 = this->vertices[face.z];

        Vector3f n1 = this->normals[face.x];
        Vector3f n2 = this->normals[face.y];
        Vector3f n3 = this->normals[face.z];
        Vector3f n = Normalize(n1 + n2 + n3);

        Interaction si = this->rayTriangleIntersect(ray, p1, p2, p3, n);
        if (si.t <= tmin && si.didIntersect) {
            siFinal = si;
            tmin = si.t;
        }
    }

    return siFinal;
}
```

The indices are for the faces, as had been discussed in class (PPT 3). The crux is that we are identifying a face using indices. `this.indices` is a vector of `Vector3<int>`.

When we have the face, we can get the vertices as described above.

We can also find the normals. These three normals are the vertex normals. (Are they?).

The final normal n is calculated using: `Vector3f n = Normalize(n1 + n2 + n3);`

`Even less Dumbo* (you)`: Do I see another ray intersect function?

`Billiman`: Yes, you do little boy/girl. And you may scene one more when we see this next function.

<b>Pay attention</b>: Here we are in our rendering journey.
```
├── scene.rayIntersect
│  ├── surface.rayIntersect
│      └── surface.rayTriangleIntersect   <-We are here. Two more steps to go.
│           └── surface.rayPlaneIntersect
```

Now we come to the `rayTriangleIntersect(ray, p1, p2, p3, n)`

This is its code:
```cpp
Interaction Surface::rayTriangleIntersect(Ray ray, Vector3f v1, Vector3f v2, Vector3f v3, Vector3f n)
{
    Interaction si = this->rayPlaneIntersect(ray, v1, n);

    if (si.didIntersect) {
        bool edge1 = false, edge2 = false, edge3 = false;

        // Check edge 1
        {
            Vector3f nIp = Cross((si.p - v1), (v3 - v1));
            Vector3f nTri = Cross((v2 - v1), (v3 - v1));
            edge1 = Dot(nIp, nTri) > 0;
        }

        // Check edge 2
        {
            Vector3f nIp = Cross((si.p - v1), (v2 - v1));
            Vector3f nTri = Cross((v3 - v1), (v2 - v1));
            edge2 = Dot(nIp, nTri) > 0;
        }

        // Check edge 3
        {
            Vector3f nIp = Cross((si.p - v2), (v3 - v2));
            Vector3f nTri = Cross((v1 - v2), (v3 - v2));
            edge3 = Dot(nIp, nTri) > 0;
        }

        if (edge1 && edge2 && edge3) {
            // Intersected triangle!
            si.didIntersect = true;
        }
        else {
            si.didIntersect = false;
        }
    }

    return si;
}
```

- First we will see if the ray intersected with a plane.
- If it did, then we would move inside the if statement. Else, would simply return.
- Remember `Iteraction` captures the following data:
```cpp
struct Interaction {
    Vector3f p, n;
    float t = 1e30f;
    bool didIntersect = false;
};
```
- Now the rest of the code is just the C++ way of telling what was told mathematically in the class. Procedures were told to calculate the interiority of intersection with a triangle.

Also, as I know you may ask this: Tell me `this->rayPlaneIntersect(ray, v1, n);`

```cpp
Interaction Surface::rayPlaneIntersect(Ray ray, Vector3f p, Vector3f n)
{
    Interaction si;

    float dDotN = Dot(ray.d, n);
    if (dDotN != 0.f) {
        float t = -Dot((ray.o - p), n) / dDotN;

        if (t >= 0.f) {
            si.didIntersect = true;
            si.t = t;
            si.n = n;
            si.p = ray.o + ray.d * si.t;
        }
    }

    return si;
}
```

Here, we are finding using past principles, the intersection of a ray to the point.

We have the equations:

$$r(t) = \vec o + t \vec d$$

$$and$$

$$(\vec x - \vec p) \cdot \vec n = 0$$

Solving for t, we get:

$$t = - \frac{(\vec o - \vec p) \cdot \vec n}{\vec d \cdot \vec n}$$

Explain this: 
```cpp
if (t >= 0.f) {
    si.didIntersect = true;
    si.t = t;
    si.n = n;
    si.p = ray.o + ray.d * si.t;
}
```

`Even less Dumbo* (you)`: What is this:

```cpp
if (si.t <= ray.t) {
    siFinal = si;
    ray.t = si.t;
}
```
\* (This applies only if you are in IIIT)

`Billiman`: We are picking up the smallest value of t that we get after the intersection. It is this value of that would be seen first. (Need more explanation?)

`No longer Dumbo (you)`: No, thank you. Can you please explain me the coloring of the code part.

`Billiman`: Sure, after all a billi should be telling it. <i>I had a billi long time back.</i>

Here is the code:
```cpp
if (si.didIntersect)
    this->outputImage.writePixelColor(0.5f * (si.n + Vector3f(1.f, 1.f, 1.f)), x, y);
else
    this->outputImage.writePixelColor(Vector3f(0.f, 0.f, 0.f), x, y);
```

Simply stated,

if an intersection happens, fill it with a color.

If not, don't fill it with a color.

Here's an explanation to how the colors are being filled:

\<put explanation here>

# Part 5: Saving the image

Now we are at the last part. We have covered everything that is required for you to understand the codebase at least for the assignments. I don't think they would ask for changes in the save image part.

For knowledge sake, let's learn this as well.

Nahi yaar mann nahi hai...

ان کي ڇڏيو

# Part 6: Concluding Thoughts

Be happy!

# Appendices

## A1: The up vector 

**(unconfirmed, need to validate with professor/TAs)**

There has been quite a lot of confusion around this. The confusion is where is the up vector? Is it the vector that describes the vector coming out of the Camera's head? Or is with respect to the scene?

If it is the latter, what do we even mean by an up vector with respect to the scene.

We'll see what this up vector really is from a lot of perspectives. It is because of so many perspectives that this confusion comes up - several people answering the same question differently with a different answer.

I remember vividly being told once or twice that the up vector is with respect to the camera. The code tells otherwise.

The fact is that people use both notions.

In particular, a vector named up **is not coming out of the head of the camera** in this codebase. In fact, it is a different vector named **u** that is doing that job.

Up appears to be some random vector over here whose purpose is to tell what is up for the scene. Then accordingly, a new "up" vector called `u` is calculated.

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fanshium.github.io%2Fcg_codebase.html&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)