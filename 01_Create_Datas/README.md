
CUDA : création du programme de calcul avec plusieurs GPU
=================

Le code cuda permet d’utiliser les GPU NVDIA comme centre de calculs.

Le code que je propose est décompose en 5 parties :

1.  Le header

C’est le code commun entre le code cuda et c++, on y trouve :

-   Le type de fractale à générer : Type\_Fractal

```c++
// Définition de l'énumération pour le type de fractale
enum Type_Fractal { Mandelbrot, Julia };
```

-   La structure **Complex** pour représenter les nombres complexes

```c++
// Définition de la structure Complex pour représenter les nombres complexes
struct Complex
{
    double x, y; // Partie réelle et imaginaire

    // Constructeur pour initialiser un nombre complexe
    __host__ __device__
    Complex(double a = 0.0, double b = 0.0) : x(a), y(b) {}

    // Surcharge de l'opérateur + pour l'addition de deux nombres complexes
    __host__ __device__
    Complex operator+(const Complex &other) const
    {
        return Complex(x + other.x, y + other.y);
    }

    // Surcharge de l'opérateur - pour la soustraction de deux nombres complexes
    __host__ __device__
    Complex operator-(const Complex &other) const
    {
        return Complex(x - other.x, y - other.y);
    }

    // Surcharge de l'opérateur * pour la multiplication de deux nombres complexes
    __host__ __device__
    Complex operator*(const Complex &other) const
    {
        return Complex(x * other.x - y * other.y, x * other.y + y * other.x);
    }

    // Fonction pour calculer la norme d'un nombre complexe
    __host__ __device__ double norm() const
    {
        return sqrt(x * x + y * y);
    }

    // Fonction pour élever un nombre complexe à une puissance donnée
    __host__ __device__
    Complex power(double p) const
    {
        double radius = sqrt(x * x + y * y);
        double angle = atan2(y, x);
        double radius_p = pow(radius, p);
        double angle_p = p * angle;

        return Complex(radius_p * cos(angle_p), radius_p * sin(angle_p));
    }
};
```

-   La structure **ParameterPicture** pour stocker les paramètres de l'image fractale

```c++
// Définition de la structure ParameterPicture pour stocker les paramètres de l'image fractale
struct ParameterPicture
{
    long lenG; // Longueur globale en 3D
    long lenL; // Longueur locale en 2D
    double2 start; // Point de départ de l'image
    double size; // Taille d'un côté de l'image
    Type_Fractal type_fractal; // Type de fractale (Mandelbrot ou Julia)
    double2 coef_julia; // Coefficients pour la fractale de Julia
    double power_value; // Valeur de la puissance
    long iter_max; // Nombre maximal d'itérations
    long id; // Identifiant de l'image

    // Constructeur pour initialiser un objet ParameterPicture
    __host__ __device__ ParameterPicture(long id, long lenG, double2 start, double size, double power_value, long iter_max, Type_Fractal type_fractal, double2 coef_julia = make_double2(0.0, 0.0)) 
        : id(id), power_value(power_value), iter_max(iter_max), type_fractal(type_fractal), coef_julia(coef_julia), lenG(lenG), lenL(floorf(sqrtf((float)lenG))), start(start), size(size) {};

    // Fonction pour obtenir la taille de l'image en 3D
    __host__ __device__ size_t Get_size_array_3D() const
    {
        return (size_t)lenG * (size_t)lenG * (size_t)lenG;
    }

    // Fonction pour obtenir la taille de l'image en 2D
    __host__ __device__ size_t Get_size_array_2D() const
    {
        return (size_t)lenG * (size_t)lenG * (size_t)lenL * (size_t)lenL;
    }

    // Fonction pour obtenir la position en coordonnées double dans l'image
    __host__ __device__ double2 GetPose_double(int x, int y, int z) const
    {
        int id = 0;
        for (long x_ = 0; x_ < lenL; x_++)
        {
            for (long y_ = 0; y_ < lenL; y_++)
            {
                if (id == z)
                {
                    return make_double2(start.x + ((double)x_ * size) + ((double)x / (double)lenG * size), start.y + ((double)y_ * size) + ((double)y / (double)lenG * size));
                }
                id++;
            }
        }
        return make_double2(0.0, 0.0);
    }

    // Fonction pour obtenir la position en coordonnées long dans l'image
    __host__ __device__ long2 GetPose_long(int x, int y, int z) const
    {
        int id = 0;
        for (long x_ = 0; x_ < lenL; x_++)
        {
            for (long y_ = 0; y_ < lenL; y_++)
            {
                if (id == z)
                {
                    return make_long2((x_ * lenG) + (long)x, (y_ * lenG) + (long)y);
                }
                id++;
            }
        }
        return make_long2(0, 0);
    }

    // Fonction pour obtenir l'index 3D d'une position dans l'image
    __host__ __device__ long Get_index_3D(int x, int y, int z) const
    {
        if (x < 0 || (long)x >= lenG)
            return -1;
        if (y < 0 || (long)y >= lenG)
            return -1;
        if (z < 0 || (long)z >= lenL * lenL)
            return -1;

        return (long)z * lenG * lenG + (long)y * lenG + (long)x;
    }

    // Fonction pour obtenir l'index 2D d'une position dans l'image
    __host__ __device__ long Get_index_2D(int x, int y, int z) const
    {
        if (x < 0 || (long)x >= lenG)
            return -1;
        if (y < 0 || (long)y >= lenG)
            return -1;
        if (z < 0 || (long)z >= (lenL * lenL))
            return -1;

        long2 pose = GetPose_long(x, y, z);
        return pose.y * lenG * lenL + pose.x;
    }

    // Fonction pour définir une valeur dans les données de l'image à une position donnée
    __host__ __device__ void Set_Value(int x, int y, int z, long *data, long value) const
    {
        long index = Get_index_2D(x, y, z);
        if (index >= 0)
        {
            data[index] = value;
        }
    }

    // Fonction pour obtenir une valeur des données de l'image à une position donnée
    __host__ __device__ long Get_Value(int x, int y, int z, long *data) const
    {
        long index = Get_index_2D(x, y, z);
        if (index >= 0)
        {
            return data[index];
        }
        else
        {
            return 0;
        }
    }

    // Fonction pour imprimer les paramètres de l'image dans un fichier
    __host__ void print_file(std::string path_file) const
    {
        std::ofstream myfile;
        myfile.open(path_file, std::ios::app);
        myfile << "id = " << id << std::endl;

        myfile << "lenG = " << lenG << std::endl;
        myfile << "lenL = " << lenL << std::endl;

        myfile << "start_x = " << start.x << std::endl;
        myfile << "start_y = " << start.y << std::endl;

        myfile << "size = " << size << std::endl;
        myfile << "type_fractal = " << type_fractal << std::endl;
        myfile << "coef_julia_x = " << coef_julia.x << std::endl;
        myfile << "coef_julia_y = " << coef_julia.y << std::endl;

        myfile << "power_value = " << power_value << std::endl;
        myfile << "iter_max = " << iter_max << std::endl;
        myfile.close();
    }
};
```

1.  Le code cuda

C’est le code qui calcul la fractale de Julia ou de Mandelbrot, on y
trouve :

-   Kernel\_Picture : Kernel CUDA pour générer une image fractale

-   RUN : la fonction pour exécuter le kernel CUDA

1.  Le code C++

C’est le code qui permet de gérer la création de fractales de Julia ou
de Mandelbrot, on y trouve :

-   File\_Generate : la structure pour gérer les fichiers (.bin et .txt)

-   RUN : Déclaration de la fonction CUDA externe

-   CreateFolder : Fonction pour créer le dossier de travail

-   if\_file\_exist : Fonction pour vérifier si un fichier existe

-   write\_bin : Fonction pour écrire des données binaires dans un
    fichier

-   run : Fonction supervision pour lancement de calculs d'une fractale

-   Get\_nbfiles\_bin : Fonction pour obtenir le nombre de fichiers
    binaires existants

-   Open\_file\_txt : Fonction pour ouvrir un fichier texte et lire son
    contenu

-   Main : Fonction principale qui est exécuté au lancement

1.  Le scripte pour compiler le programme.

C’est le scripte qui permet de générer l’application

1.  Les paramètres

C’est les paramètres de calculs externes au programme, on y trouve :

-   L’id de la care nvdia à utiliser de 0 à N, n étant le nombre -1 de
    cartes graphiques disponibles

-   La borne minimale du coef y de Julia

-   La borne maximale du coef y de Julia
