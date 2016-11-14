#include <stdio.h>
#include <stdlib.h>
#include <string>

using namespace std;

int main(int argc, char**argv)
{
	if(argc != 2)
	{
		printf("usage: ./transform file_name\n");
		exit(0);
	}

	string file_name(argv[1]);

	FILE* file_in = fopen(file_name.c_str(), "r");
	if(file_in == NULL)
	{
		printf("usage: ./transform file_name\n");
		exit(0);
	}

	//transform
	FILE* file_out = fopen((file_name + ".bin").c_str(), "wb");


	char line[256];

	//abandon the file headers
	fscanf(file_in, "%[^\n]\n", line);
	fscanf(file_in, "%[^\n]\n", line);
	fscanf(file_in, "%[^\n]\n", line);
	
	int u,v;
	float r;
	while (fscanf(file_in, "%d %d %f", &u, &v, &r) > 0) 
	{
		fwrite(&u, sizeof(int), 1,  file_out);
       	fwrite(&v, sizeof(int), 1,  file_out);
       	fwrite(&r, sizeof(float), 1,file_out);
	}

	fclose(file_in);
	fclose(file_out);

	return 0;
}