# Dev-oriented image: runs the Vite dev server with hot reload against a
# volume-mounted source tree (see docker-compose.yml). For a production
# artifact, build separately with `npm run build` and serve `dist/` behind
# nginx/serve — not needed for this prototype's docker-compose workflow.
FROM node:20-slim

WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host"]
