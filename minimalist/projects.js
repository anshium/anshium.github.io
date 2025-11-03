const ProjectCategory = {
  ROBOTICS: 'Robotics',
  VISION: 'Computer Vision',
  SYSTEMS: 'Systems Programming',
};

const mockProjects = [
  {
    id: 1,
    title: 'Autonomous Navigation Bot',
    description: 'A wheeled robot using SLAM (Simultaneous Localization and Mapping) with a LiDAR sensor to navigate and map unknown environments. Built with ROS and Python.',
    imageUrl: 'https://picsum.photos/seed/robotics1/600/400',
    tags: ['ROS', 'Python', 'SLAM', 'LiDAR'],
    category: ProjectCategory.ROBOTICS,
  },
  {
    id: 2,
    title: 'Object Detection for Retail',
    description: 'A deep learning model based on YOLOv8 that identifies products on store shelves to automate inventory management. Trained on a custom dataset for high accuracy.',
    imageUrl: 'https://picsum.photos/seed/vision1/600/400',
    tags: ['PyTorch', 'YOLOv8', 'OpenCV', 'AI'],
    category: ProjectCategory.VISION,
  },
  {
    id: 3,
    title: 'Custom Memory Allocator',
    description: 'A high-performance memory allocator written in C++ designed to reduce fragmentation and improve allocation speed for multi-threaded applications.',
    imageUrl: 'https://picsum.photos/seed/systems1/600/400',
    tags: ['C++', 'Memory Management', 'Multithreading'],
    category: ProjectCategory.SYSTEMS,
  },
  {
    id: 4,
    title: 'Robotic Arm Control System',
    description: 'Inverse kinematics solver and motion planning for a 6-DOF robotic arm. Implemented a user interface to control the arm in real-time.',
    imageUrl: 'https://picsum.photos/seed/robotics2/600/400',
    tags: ['C++', 'Robotics', 'Kinematics'],
    category: ProjectCategory.ROBOTICS,
  },
  {
    id: 5,
    title: 'Real-time Facial Recognition',
    description: 'Developed a system that performs real-time facial recognition from a video stream using Haar Cascades and a Siamese Network for verification.',
    imageUrl: 'https://picsum.photos/seed/vision2/600/400',
    tags: ['Python', 'OpenCV', 'TensorFlow', 'AI'],
    category: ProjectCategory.VISION,
  },
  {
    id: 6,
    title: 'Lightweight Kernel Module',
    description: 'A simple Linux kernel module written in C that creates a new /proc file to report system information. A deep dive into kernel-level programming.',
    imageUrl: 'https://picsum.photos/seed/systems2/600/400',
    tags: ['C', 'Linux', 'Kernel Development'],
    category: ProjectCategory.SYSTEMS,
  },
  {
    id: 7,
    title: 'Swarm Robotics Simulation',
    description: 'A simulation environment built in Gazebo for testing collective behavior algorithms on a swarm of miniature drones, focusing on emergent patterns.',
    imageUrl: 'https://picsum.photos/seed/robotics3/600/400',
    tags: ['Gazebo', 'ROS', 'Simulation', 'Swarm AI'],
    category: ProjectCategory.ROBOTICS,
  },
  {
    id: 8,
    title: 'Image Segmentation for Medical Scans',
    description: 'Utilized a U-Net architecture to segment tumors in MRI scans. The model achieves high precision and recall, aiding in diagnostic processes.',
    imageUrl: 'https://picsum.photos/seed/vision3/600/400',
    tags: ['PyTorch', 'U-Net', 'Medical Imaging', 'AI'],
    category: ProjectCategory.VISION,
  },
];

/**
 * Creates the HTML for a single project card.
 * @param {object} project - The project data object.
 * @returns {string} The HTML string for the project card.
 */
function createProjectCardHTML(project) {
  const MAX_DESC_LENGTH = 140;
  const truncatedDescription = project.description.length > MAX_DESC_LENGTH
    ? `${project.description.substring(0, MAX_DESC_LENGTH)}...`
    : project.description;

  const tagsHTML = project.tags.map(tag => `
    <span class="inline-block bg-accent text-highlight rounded-full px-3 py-1 text-sm font-semibold hover:bg-highlight hover:text-text-primary transition-colors duration-200 cursor-pointer">
      #${tag}
    </span>
  `).join('');

  return `
    <div class="bg-secondary rounded-xl overflow-hidden shadow-lg hover:shadow-2xl transform hover:-translate-y-2 transition-all duration-300 ease-in-out flex flex-col h-full border border-accent">
      <div class="relative">
        <img class="w-full h-56 object-cover" src="${project.imageUrl}" alt="${project.title}" />
        <div class="absolute top-0 right-0 bg-highlight text-text-primary px-3 py-1 m-2 rounded-full text-xs font-bold">
          ${project.category}
        </div>
      </div>
      <div class="p-6 flex flex-col flex-grow">
        <h3 class="font-bold text-xl mb-2 text-text-primary">${project.title}</h3>
        <p class="text-text-secondary text-base flex-grow mb-4">
          ${truncatedDescription}
        </p>
        <div class="flex flex-wrap gap-2 pt-4 border-t border-accent">
          ${tagsHTML}
        </div>
      </div>
    </div>
  `;
}

/**
 * Creates the HTML for a category section with its project cards.
 * @param {string} title - The title of the project category.
 * @param {object[]} projects - An array of project objects in that category.
 * @returns {string} The HTML string for the entire section.
 */
function createProjectListHTML(title, projects) {
  if (projects.length === 0) {
    return '';
  }

  const projectCardsHTML = projects.map(createProjectCardHTML).join('');

  return `
    <section>
      <h2 class="text-3xl font-bold text-text-primary mb-8 border-l-4 border-highlight pl-4">
        ${title}
      </h2>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        ${projectCardsHTML}
      </div>
    </section>
  `;
}

/**
 * Main function to render all project lists into the DOM.
 */
function renderProjects() {
  const projectContainer = document.getElementById('project-container');
  if (!projectContainer) {
    console.error('Project container not found!');
    return;
  }

  const roboticsProjects = mockProjects.filter(p => p.category === ProjectCategory.ROBOTICS);
  const visionProjects = mockProjects.filter(p => p.category === ProjectCategory.VISION);
  const systemsProjects = mockProjects.filter(p => p.category === ProjectCategory.SYSTEMS);

  let fullHTML = '';
  fullHTML += createProjectListHTML(ProjectCategory.ROBOTICS, roboticsProjects);
  fullHTML += createProjectListHTML(ProjectCategory.VISION, visionProjects);
  fullHTML += createProjectListHTML(ProjectCategory.SYSTEMS, systemsProjects);
  
  projectContainer.innerHTML = fullHTML;
}

// Wait for the DOM to be fully loaded before running the script
document.addEventListener('DOMContentLoaded', renderProjects);