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

### Part 1.1: Overview of the thingy called Scene

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

- First, this `struct Scene` has a vector of structs `Surface` names surfaces.
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

Now how there surfaces are created and what is the `Vector2i imageResolution` and more things, we will see later on in the following sections. ::Improve this line

### Part 1.2: Overview of the thingy called Integrator

Now our focus is on this line of code:
`Integrator rayTracer(scene);`

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
- It also has a Texture struct inside it. The texture struct is named outputImage.

There is also this cute little render() function which is very long. <i>So long that `long` had to be written twice. </i>

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

Not much of a big deal. It is more of terminogolies and simple math here:

See this and get afraid:
```cpp
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

Now comes an interesting part. Having a biscuit, if you like.

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

## Part 3.2: createSurfaces

createSurfaces just turns out to be a good 125 line code in itself. I'll make a video about it probably. It would take too much of a text to explain it here. Or I'll see what's best.

After this we have:

`this->surfaces.insert(this->surfaces.end(), surf.begin(), surf.end());`

This line simply adds the surf obtained to the end of the vector `surfaces` in the `struct Scene`

This is how we increment the iterator: `surfaceIdx = surfaceIdx + surf.size();`

## Part 4: Rendering

### Part 4.1: Introduction

Ah, tis the part that most of us are exited about, like an electron. <i>I might soon release a form to rate my jokes.</i>

When we are done with initialising the integrator part, we call the render() function like this:

`auto renderTime = rayTracer.render();`

For now, we'll ignore the renderTime part.

We had previously made a struct `ratTracer` of the type `Integrator` and have initialised it with the `scene`. The scene's components were populated using the steps mentioned previously.

Now for the most part, here is the render function:
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

For each pixel, generate a ray from that pixel. Which pixel are we talking about, you may ask. Well, it is the pixels we have in our camera.

How many pixels are there. Well this depends on the `image resolution` as it is very clear from the two `for loops`.

Then we see if that ray intersects with the scene.

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

The pixel center is calculated from the upperLeft point (remember it? Just scroll up to find out) and the pixelDeltaU/V. This is simply to calulate the offset of the pixelCenter based on the camera. I understand it. If you don't, let me know I'll write more here about it here (or even make a video).

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

There are two Vector3fs in here: p and n. p is position. n is normal.
Position is where the interaction happened. n is the normal of the surface at that point of interaction.

Honestly, even I don't know what `float t = 1e30f;` is.

Then there is information if the intersection did not happened using, `didIntersect` which is by default initialised to false.

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

- Here we have an Interaction siFinal. 
- We go over all the surfaces we have.
- For each surface, we have an Interaction si (surface interaction)
- This Interaction si is computed using another function of the same name rayIntersect.
- This checks the intersection of the ray with a particular scene and not just a surface in particular.

The surface.rayIntersect(ray) is as follows:

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

`Even less Dumbo* (you)`:

\* (This applies only if you are in IIIT)

What is this:
```cpp
if (si.t <= ray.t) {
    siFinal = si;
    ray.t = si.t;
}
```

`Billiman`: Here we see the first use of the large value `t` we had in the `struct Interaction`. Get the flashbacks:
`float t = 1e30f;`

